import os
import lmdb
import msgpack
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Union, List, Tuple
import time
import argparse
import sys

# 添加CLIP模型路径
sys.path.append('/home/stu_b/BIKE')
import clip
from improved_qwen_clip_generator import CustomQwen2VLForConditionalGeneration
class QwenCLIPLMDBGenerator:
    def __init__(self, frames_per_video=5, gpu_ids=None, clip_config=None):
        """
        结合CLIP和Qwen的LMDB I-frame描述生成器
        
        Args:
            frames_per_video: 每个视频采样的帧数
            gpu_ids: 指定使用的GPU ID列表
            clip_config: CLIP模型配置字典
        """
        # GPU设置
        self.gpu_ids = gpu_ids if gpu_ids is not None else []
        self.setup_gpu_environment()
        
        self.frames_per_video = frames_per_video
        print(f"🚀 使用设备: {self.device}")
        print(f"🎮 GPU设置: {self.gpu_ids}")
        print(f"🎬 每个视频采样帧数: {self.frames_per_video}")
        
        # CLIP配置
        self.clip_config = clip_config or {
            'arch': 'ViT-B/32',
            'embed_dim': 512,  # ViT-B: 512, ViT-L: 768
            'residual_layers_to_use': [0, 11],
            'mvs_layers_to_use': [0, 11]
        }
        
        # 初始化时不加载模型
        self.clip_model = None
        self.qwen_model = None
        self.qwen_processor = None
        self.models_loaded = False
        
        # 创建图像保存目录
        self.image_save_dir = "/home/stu_b/BIKE/hmdb_qwen"
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"📁 图像保存目录: {self.image_save_dir}")
        
        print("✅ CLIP+Qwen专用LMDB I-frame描述生成器初始化完成")
    
    def setup_gpu_environment(self):
        """设置GPU环境"""
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将使用CPU")
            self.device = torch.device("cpu")
            self.gpu_ids = []
            return
        
        total_gpus = torch.cuda.device_count()
        print(f"🎮 系统可用GPU数量: {total_gpus}")
        
        if not self.gpu_ids:
            self.gpu_ids = [0] if total_gpus > 0 else []
            print(f"🎮 未指定GPU，自动使用: GPU {self.gpu_ids}")
        else:
            valid_gpu_ids = []
            for gpu_id in self.gpu_ids:
                if 0 <= gpu_id < total_gpus:
                    valid_gpu_ids.append(gpu_id)
                    print(f"✅ GPU {gpu_id} 可用")
                else:
                    print(f"❌ GPU {gpu_id} 不可用 (系统只有 {total_gpus} 个GPU)")
            self.gpu_ids = valid_gpu_ids
        
        if self.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
            print(f"🎮 设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"🎮 GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ 没有可用的GPU，将使用CPU")
            self.device = torch.device("cpu")
    
    def load_clip_model(self):
        """加载CLIP模型 - 智能权重适配"""
        print("🔧 正在加载CLIP模型...")
        try:
            # 加载基础模型
            self.clip_model, clip_state_dict = clip.load(
                self.clip_config['arch'],
                device='cpu',
                jit=False,
                internal_modeling=None,
                Block="Origin",
                T=self.frames_per_video,
                dropout=0.,
                emb_dropout=0.,
                pretrain=None,
                joint_st=False,
                residual_layers_to_use=self.clip_config['residual_layers_to_use'],
                mvs_layers_to_use=self.clip_config['mvs_layers_to_use']
            )
            
            print("✅ CLIP基础模型加载成功")
            
            # 加载和适配预训练权重
            pretrained_path = "/home/stu_b/BIKE/exps/hmdb51/ViT-B/16/I_mv_res_全训练结果_76.6/model_best.pt"
            
            if os.path.exists(pretrained_path):
                print(f"🔧 正在加载预训练权重: {pretrained_path}")
                
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                pretrained_state_dict = checkpoint['model_state_dict']
                
                # 适配权重
                adapted_state_dict = self.adapt_pretrained_weights(
                    pretrained_state_dict, 
                    self.clip_model.state_dict()
                )
                
                # 加载适配后的权重
                missing_keys, unexpected_keys = self.clip_model.load_state_dict(
                    adapted_state_dict, strict=False
                )
                
                print(f"✅ 权重适配完成 - 缺失: {len(missing_keys)}, 意外: {len(unexpected_keys)}")
                
                if 'epoch' in checkpoint:
                    print(f"🔧 预训练模型训练轮数: {checkpoint['epoch']}")
            
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            return True
            
        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            return False

    def adapt_pretrained_weights(self, pretrained_dict, current_dict):
        """适配预训练权重到当前模型"""
        adapted_dict = {}
        
        for key, current_param in current_dict.items():
            if key in pretrained_dict:
                pretrained_param = pretrained_dict[key]
                
                if current_param.shape == pretrained_param.shape:
                    # 形状匹配，直接使用
                    adapted_dict[key] = pretrained_param
                    print(f"✅ {key}: 直接匹配")
                else:
                    # 形状不匹配，尝试适配
                    adapted_param = self.adapt_parameter(
                        pretrained_param, current_param, key
                    )
                    if adapted_param is not None:
                        adapted_dict[key] = adapted_param
                        print(f"🔧 {key}: 适配成功 {pretrained_param.shape} -> {current_param.shape}")
                    else:
                        print(f"⚠️ {key}: 无法适配 {pretrained_param.shape} -> {current_param.shape}")
            else:
                print(f"⚠️ {key}: 预训练权重中不存在")
        
        return adapted_dict

    def adapt_parameter(self, pretrained_param, current_param, param_name):
        """适配单个参数"""
        try:
            if 'positional_embedding' in param_name:
                # 位置编码适配
                return self.adapt_positional_embedding(pretrained_param, current_param)
            elif 'conv1.weight' in param_name:
                # 卷积权重适配
                return self.adapt_conv_weight(pretrained_param, current_param)
            else:
                # 其他参数暂时跳过
                return None
        except Exception as e:
            print(f"❌ 参数适配失败 {param_name}: {e}")
            return None

    def adapt_positional_embedding(self, pretrained_pos, current_pos):
        """适配位置编码"""
        # pretrained: [197, 768] -> current: [50, 768]
        # 简单的截取或插值策略
        
        if pretrained_pos.shape[0] > current_pos.shape[0]:
            # 截取策略：保留前N个位置
            print(f"🔧 位置编码截取: {pretrained_pos.shape[0]} -> {current_pos.shape[0]}")
            return pretrained_pos[:current_pos.shape[0]]
        elif pretrained_pos.shape[0] < current_pos.shape[0]:
            # 插值策略
            print(f"🔧 位置编码插值: {pretrained_pos.shape[0]} -> {current_pos.shape[0]}")
            # 这里可以实现更复杂的插值
            return current_pos  # 暂时使用当前权重
        else:
            return pretrained_pos

    def adapt_conv_weight(self, pretrained_conv, current_conv):
        """适配卷积权重"""
        # pretrained: [768, 3, 16, 16] -> current: [768, 3, 32, 32]
        # 可以使用插值或其他策略
        
        print(f"🔧 卷积权重适配: {pretrained_conv.shape} -> {current_conv.shape}")
        
        if pretrained_conv.shape[2:] != current_conv.shape[2:]:
            # 使用双线性插值调整kernel size
            import torch.nn.functional as F
            
            # 将权重reshape为适合插值的形状
            pretrained_reshaped = pretrained_conv.view(-1, 1, *pretrained_conv.shape[2:])
            target_size = current_conv.shape[2:]
            
            adapted_reshaped = F.interpolate(
                pretrained_reshaped, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            adapted_conv = adapted_reshaped.view(*current_conv.shape)
            return adapted_conv
        
        return pretrained_conv

    
    def setup_hf_mirror(self):
        """设置Hugging Face镜像源"""
        print("🔧 配置Hugging Face镜像源...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/stu_b/.cache/huggingface'
        cache_dir = '/home/stu_b/.cache/huggingface'
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✅ 镜像源设置完成: {os.environ.get('HF_ENDPOINT')}")
        print(f"✅ 缓存目录: {cache_dir}")
    
    def load_qwen_model(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        """加载Qwen模型"""
        print(f"🔧 正在加载Qwen模型: {model_name}")
        self.setup_hf_mirror()
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
            
            # 加载处理器
            self.qwen_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir='/home/stu_b/.cache/huggingface'
            )
            print("✅ Qwen处理器加载成功")
            
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir='/home/stu_b/.cache/huggingface'
            )

            # 使用自定义模型类
            if torch.cuda.is_available() and self.gpu_ids:
                if len(self.gpu_ids) > 1:
                    device_map = "auto"
                else:
                    device_map = {"": self.gpu_ids[0]}
                
                self.qwen_model = CustomQwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    cache_dir='/home/stu_b/.cache/huggingface'
                )
            else:
                self.qwen_model = CustomQwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir='/home/stu_b/.cache/huggingface'
                )
                self.qwen_model.to(self.device)
            


            # 🔧 强制移动投影层到主设备
            main_device = next(self.qwen_model.parameters()).device
            print(f"🔧 Qwen模型主设备: {main_device}")
            
            # 检查投影层设备
            projector_device = next(self.qwen_model.clip_feature_projector.parameters()).device
            print(f"🔧 投影层初始设备: {projector_device}")
            
            if projector_device != main_device:
                print(f"🔧 移动投影层从 {projector_device} 到 {main_device}")
                self.qwen_model.clip_feature_projector = self.qwen_model.clip_feature_projector.to(main_device)
                print(f"🔧 投影层移动后设备: {next(self.qwen_model.clip_feature_projector.parameters()).device}")
            self.qwen_model.eval()
            print("✅ Qwen模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ Qwen模型加载失败: {e}")
            return False
    
    def initialize_models(self):
        """初始化所有模型"""
        print("🎯 === 初始化模型 ===")
        
        # 加载CLIP模型
        if not self.load_clip_model():
            print("💥 CLIP模型加载失败")
            return False
        
        # 加载Qwen模型
        if not self.load_qwen_model():
            print("💥 Qwen模型加载失败")
            return False
        
        self.models_loaded = True
        print("🎉 所有模型加载成功！")
        return True
    
    def open_lmdb_database(self, lmdb_path: str):
        """打开LMDB数据库"""
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB数据库不存在: {lmdb_path}")
        
        env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                       readonly=True, lock=False,
                       readahead=False, meminit=False)
        
        print(f"📂 成功打开LMDB数据库: {lmdb_path}")
        return env
    
    def uniform_sample_frames(self, total_frames: int, num_samples: int) -> List[int]:
        """均匀采样帧索引"""
        if total_frames <= num_samples:
            return list(range(total_frames))
        
        indices = []
        step = total_frames / num_samples
        for i in range(num_samples):
            idx = int(i * step)
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        
        indices = sorted(list(set(indices)))
        print(f"🎯 从 {total_frames} 帧中均匀采样 {len(indices)} 帧: {indices}")
        return indices
    
    def extract_multimodal_frames_from_lmdb(self, image_lmdb_path: str, res_lmdb_path: str, 
                                        mv_lmdb_path: str, order_key: str, 
                                        num_frames: int = None) -> Tuple[List[Image.Image], List[Image.Image], List[np.ndarray]]:
        """
        从多个LMDB提取图像、残差和运动向量数据
        
        Args:
            image_lmdb_path: 图像LMDB路径
            res_lmdb_path: 残差LMDB路径
            mv_lmdb_path: 运动向量LMDB路径
            order_key: 视频键
            num_frames: 采样帧数
        
        Returns:
            (images, residuals, motion_vectors)
        """
        if num_frames is None:
            num_frames = self.frames_per_video
        
        # 打开三个数据库
        image_env = self.open_lmdb_database(image_lmdb_path)
        res_env = self.open_lmdb_database(res_lmdb_path)
        mv_env = self.open_lmdb_database(mv_lmdb_path)
        
        images, residuals, motion_vectors = [], [], []
        
        def get_video_data_from_lmdb(env, order_key):
            """
            从LMDB中获取视频数据的通用方法
            """
            with env.begin(write=False) as txn:
                try:
                    # 首先尝试使用__order__和__keys__映射
                    order_data = msgpack.loads(txn.get(b'__order__'))
                    keys_data = msgpack.loads(txn.get(b'__keys__'))
                    
                    if order_key in order_data:
                        order_index = order_data.index(order_key)
                        actual_key = keys_data[order_index]
                        video_data = txn.get(actual_key)
                        print(f"🔍 通过__order__映射找到数据: {order_key} -> {actual_key}")
                        return video_data
                    else:
                        print(f"🔍 在__order__中未找到 {order_key}，尝试直接访问")
                        # 如果在order中找不到，尝试直接访问
                        video_data = txn.get(order_key.encode() if isinstance(order_key, str) else order_key)
                        return video_data
                        
                except Exception as e:
                    print(f"🔍 __order__/__keys__处理失败: {e}，尝试直接访问")
                    # 如果__order__/__keys__处理失败，尝试直接访问
                    video_data = txn.get(order_key.encode() if isinstance(order_key, str) else order_key)
                    return video_data
        
        try:
            # 从图像数据库获取采样索引
            print(f"🔍 从图像LMDB获取帧索引...")
            image_video_data = get_video_data_from_lmdb(image_env, order_key)
            
            if image_video_data is None:
                raise ValueError(f"未找到图像数据: {order_key}")
            
            frames_data = msgpack.loads(image_video_data, raw=True)
            total_frames = len(frames_data)
            print(f"📊 视频 {order_key} 总帧数: {total_frames}")
            
            # 获取采样索引
            frame_indices = self.uniform_sample_frames(total_frames, num_frames)
            
            # 提取图像帧
            print(f"🔍 提取图像帧...")
            for i, frame_idx in enumerate(frame_indices):
                try:
                    frame_bytes = frames_data[frame_idx]
                    image = Image.open(BytesIO(frame_bytes)).convert('RGB')
                    images.append(image)
                    print(f"✅ 提取图像帧 {i+1}/{len(frame_indices)} (索引{frame_idx}): {image.size}")
                except Exception as e:
                    print(f"❌ 提取图像帧 {frame_idx} 失败: {e}")
            
            # 提取残差帧
            print(f"🔍 提取残差帧...")
            res_video_data = get_video_data_from_lmdb(res_env, order_key)
            
            if res_video_data:
                try:
                    res_frames_data = msgpack.loads(res_video_data, raw=True)
                    print(f"📊 残差数据帧数: {len(res_frames_data)}")
                    
                    for i, frame_idx in enumerate(frame_indices):
                        try:
                            if frame_idx < len(res_frames_data):
                                frame_bytes = res_frames_data[frame_idx]
                                residual = Image.open(BytesIO(frame_bytes)).convert('RGB')
                                residuals.append(residual)
                                print(f"✅ 提取残差帧 {i+1}/{len(frame_indices)} (索引{frame_idx}): {residual.size}")
                            else:
                                print(f"⚠️ 残差帧索引 {frame_idx} 超出范围，创建零残差")
                                if images:
                                    zero_residual = Image.new('RGB', images[i].size, (128, 128, 128))
                                    residuals.append(zero_residual)
                        except Exception as e:
                            print(f"❌ 提取残差帧 {frame_idx} 失败: {e}")
                            # 如果残差提取失败，创建零残差
                            if images:
                                zero_residual = Image.new('RGB', images[i].size, (128, 128, 128))
                                residuals.append(zero_residual)
                except Exception as e:
                    print(f"❌ 残差数据解析失败: {e}")
                    # 为所有帧创建零残差
                    for i in range(len(images)):
                        zero_residual = Image.new('RGB', images[i].size, (128, 128, 128))
                        residuals.append(zero_residual)
            else:
                print(f"⚠️ 未找到残差数据，创建零残差")
                # 为所有帧创建零残差
                for i in range(len(images)):
                    zero_residual = Image.new('RGB', images[i].size, (128, 128, 128))
                    residuals.append(zero_residual)
            # 提取运动向量帧
            print(f"🔍 提取运动向量帧...")
            mv_video_data = get_video_data_from_lmdb(mv_env, order_key)
            if mv_video_data:
                try:
                    mv_video_data = msgpack.loads(mv_video_data, raw=True)
                    print(f"📊 运动向量数据帧数: {len(mv_video_data)}")
                    
                    for i, frame_idx in enumerate(frame_indices):
                        try:
                            if frame_idx < len(mv_video_data):
                                frame_bytes = mv_video_data[frame_idx]
                                mv = Image.open(BytesIO(frame_bytes)).convert('RGB')
                                motion_vectors.append(mv)
                                print(f"✅ 提取运动向量帧 {i+1}/{len(frame_indices)} (索引{frame_idx}): {mv.size}")
                            else:
                                print(f"⚠️ 运动向量帧索引 {frame_idx} 超出范围，创建零运动向量")
                                if images:
                                    zero_mv = Image.new('RGB', images[i].size, (128, 128, 128))
                                    motion_vectors.append(zero_mv)
                        except Exception as e:
                            print(f"❌ 提取运动向量帧 {frame_idx} 失败: {e}")
                            # 如果运动向量提取失败，创建零运动向量
                            if images:
                                zero_mv = Image.new('RGB', images[i].size, (128, 128, 128))
                                motion_vectors.append(zero_mv)
                except Exception as e:
                    print(f"❌ 运动向量数据解析失败: {e}")
                    # 为所有帧创建零运动向量
                    for i in range(len(images)):
                        zero_mv = Image.new('RGB', images[i].size, (128, 128, 128))
                        motion_vectors.append(zero_mv)
            else:
                print(f"⚠️ 未找到运动向量数据，创建零运动向量")
                # 为所有帧创建零运动向量
                for i in range(len(images)):
                    zero_mv = Image.new('RGB', images[i].size, (128, 128, 128))
                    motion_vectors.append(zero_mv)
        
        finally:
            image_env.close()
            res_env.close()
            mv_env.close()
        
        print(f"🎬 成功提取: 图像{len(images)}帧, 残差{len(residuals)}帧, 运动向量{len(motion_vectors)}帧")
        return images, residuals, motion_vectors


    def preprocess_multimodal_data(self, images: List[Image.Image], 
                                 residuals: List[Image.Image], 
                                 motion_vectors: List[np.ndarray],
                                 target_size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预处理多模态数据为tensor
        
        Args:
            images: 图像列表
            residuals: 残差列表
            motion_vectors: 运动向量列表
            target_size: 目标尺寸
        
        Returns:
            (image_tensor, residual_tensor, mv_tensor)
        """
        # 预处理图像
        image_tensors = []
        for img in images:
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # CHW
            image_tensors.append(img_tensor)
        
        # 预处理残差
        residual_tensors = []
        for res in residuals:
            res_resized = res.resize(target_size, Image.LANCZOS)
            res_array = np.array(res_resized).astype(np.float32) / 255.0
            res_tensor = torch.from_numpy(res_array).permute(2, 0, 1)  # CHW
            residual_tensors.append(res_tensor)
        
        # 预处理运动向量
        mv_tensors = []
        for mv in motion_vectors:
            mv_resized = res.resize(target_size, Image.LANCZOS)
            mv_array = np.array(mv_resized).astype(np.float32) / 255.0
            mv_tensor = torch.from_numpy(mv_array).permute(2, 0, 1)  # CHW
            mv_tensors.append(mv_tensor)
        
        # 堆叠为批次张量
        image_batch = torch.stack(image_tensors, dim=0)  # [T, C, H, W]
        residual_batch = torch.stack(residual_tensors, dim=0)  # [T, C, H, W]
        mv_batch = torch.stack(mv_tensors, dim=0)  # [T, C, H, W]
        
        print(f"🔧 预处理完成: 图像{image_batch.shape}, 残差{residual_batch.shape}, 运动向量{mv_batch.shape}")
        return image_batch, residual_batch, mv_batch
    
    def encode_with_clip(self, image_batch: torch.Tensor, 
                        residual_batch: torch.Tensor, 
                        mv_batch: torch.Tensor) -> torch.Tensor:
        """
        使用CLIP模型编码多模态特征
        
        Args:
            image_batch: [T, C, H, W]
            residual_batch: [T, C, H, W]
            mv_batch: [T, C, H, W]
        
        Returns:
            merged_features: [T, embed_dim]
        """
        print("🤖 使用CLIP编码多模态特征...")
        
        with torch.no_grad():
            # 移动到设备
            image_batch = image_batch.to(self.device)
            residual_batch = residual_batch.to(self.device)
            mv_batch = mv_batch.to(self.device)
            
            # 使用CLIP的encode_image方法
            # 注意：你的encode_image期望的输入格式是 [b*t, c, h, w]
            b, t, c, h, w = 1, image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], image_batch.shape[3]
            
            # Reshape为 [b*t, c, h, w] 格式
            image_input = image_batch.view(-1, c, h, w)  # [T, C, H, W]
            residual_input = residual_batch.view(-1, c, h, w)  # [T, C, H, W]
            mv_input = mv_batch.view(-1, mv_batch.shape[1], h, w)  # [T, C_mv, H, W]
            
            # 使用CLIP编码
            image_features, res_features, mvs_features = self.clip_model.encode_image(image_input, residual_input, mv_input)
            # merged_features shape: [T, embed_dim]
            merged_features = image_features + res_features + mvs_features 
            # 转换为与Qwen模型匹配的数据类型
            if torch.cuda.is_available() and self.qwen_model is not None:
                # 检查Qwen模型的数据类型
                qwen_dtype = next(self.qwen_model.parameters()).dtype
                merged_features = merged_features.to(dtype=qwen_dtype)
                print(f"🔧 CLIP特征数据类型转换: {merged_features.dtype}")
            
            print(f"🔧 CLIP编码完成: {merged_features.shape}")
            return merged_features
    
    def construct_qwen_inputs_with_clip_features(self, clip_features: torch.Tensor, 
                                            prompt: str) -> dict:
        """
        使用CLIP特征构建Qwen模型的输入
        
        Args:
            clip_features: CLIP编码的特征 [T, embed_dim] - 整个视频的特征
            prompt: 文本提示
        
        Returns:
            inputs: 兼容Qwen模型的输入字典
        """
        print("🔧 构建Qwen模型输入...")
        
        # 1. 将多帧特征聚合为单个视频特征
        # 可以使用平均池化、最大池化或者注意力机制
        video_feature = clip_features.mean(dim=0, keepdim=True)  # [1, embed_dim]
        print(f"🔧 视频特征聚合: {clip_features.shape} -> {video_feature.shape}")
        
        # 2. 构建文本prompt（只包含一个图像token，代表整个视频）
        # 注意：这里只用一个图像token，因为我们把整个视频当作一个"图像"
        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        
        full_prompt = f"This is a video represented as a single feature encoding. {prompt}"
        text_prompt = f"<|im_start|>user\n{image_token}{full_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 3. 使用Qwen processor处理文本（不包含图像）
        text_inputs = self.qwen_processor(
            text=[text_prompt],
            images=None,  # 不传入图像
            return_tensors="pt",
            padding=True
        )
        
        # 4. 构建兼容的pixel_values
        embed_dim = video_feature.shape[1]
        
        # 将视频特征扩展到合适的维度
        if embed_dim < 1176:
            padding_size = 1176 - embed_dim
            padding = torch.zeros(1, padding_size, device=video_feature.device, dtype=video_feature.dtype)
            pixel_values = torch.cat([video_feature, padding], dim=1)
        elif embed_dim > 1176:
            pixel_values = video_feature[:, :1176]
        else:
            pixel_values = video_feature
        
        # 扩展到目标patch数量 (Qwen期望的格式)
        # 对于单个视频特征，我们需要重复或插值到合适的patch数量
        target_patches = 1024  # 调整这个数字以匹配Qwen的期望
        current_patches = pixel_values.shape[0]
        
        if current_patches < target_patches:
            # 重复特征到目标数量
            repeat_times = target_patches // current_patches
            remainder = target_patches % current_patches
            
            repeated_features = pixel_values.repeat(repeat_times, 1)
            if remainder > 0:
                repeated_features = torch.cat([repeated_features, pixel_values[:remainder]], dim=0)
            pixel_values = repeated_features
        else:
            pixel_values = pixel_values[:target_patches]
        
        # 5. 构建image_grid_thw (单个"图像"的网格信息)
        # 这里我们告诉Qwen这是一个32x32的图像网格
        image_grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.int64)
        
        # 6. 构建最终输入
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw
        }
        
        print(f"🔧 Qwen输入构建完成:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
        
        return inputs
    
    def generate_qwen_description_with_clip(self, images: List[Image.Image], 
                                        residuals: List[Image.Image],
                                        motion_vectors: List[np.ndarray],
                                        prompt: str = "Please describe what is happening in this video. Focus on the actions, objects, and scene details you can observe.") -> str:
        """
        使用CLIP特征+Qwen生成视频描述
        """
        print(f"🤖 正在使用CLIP+Qwen分析 {len(images)} 帧视频...")
        
        try:
            # 🔧 获取模型的主设备
            model_device = next(self.qwen_model.parameters()).device
            print(f"🔧 Qwen模型主设备: {model_device}")
            
            # 1. 预处理多模态数据
            image_batch, residual_batch, mv_batch = self.preprocess_multimodal_data(
                images, residuals, motion_vectors
            )
            
            # 2. 使用CLIP编码特征
            clip_features = self.encode_with_clip(image_batch, residual_batch, mv_batch)
            
            # 3. 聚合特征
            aggregated_features = clip_features.mean(dim=0, keepdim=True)  # [1, 512]
            
            # 4. 确保数据类型和设备匹配
            qwen_dtype = next(self.qwen_model.parameters()).dtype
            aggregated_features = aggregated_features.to(dtype=qwen_dtype, device=model_device)
            print(f"🔧 聚合特征设备: {aggregated_features.device}, 类型: {aggregated_features.dtype}")
            
            # 5. 构建文本输入
            text_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            text_inputs = self.qwen_processor(text=[text_prompt], images=None, return_tensors="pt")
            
            # 🔧 确保文本输入在正确设备上
            for key in text_inputs:
                text_inputs[key] = text_inputs[key].to(model_device)
            
            # 6. 使用自定义模型生成
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    clip_features=aggregated_features,  # 已经在正确设备上
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.8,
                    pad_token_id=self.qwen_processor.tokenizer.eos_token_id
                )
            
            # 7. 解码
            input_length = text_inputs['input_ids'].shape[1]
            new_tokens = generated_ids[0][input_length:]
            
            output_text = self.qwen_processor.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True
            )
            
            return output_text.strip() if output_text.strip() else "生成的视频描述为空"
            
        except Exception as e:
            error_msg = f"❌ CLIP+Qwen视频描述生成失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg


    
    def get_video_keys_from_order(self, lmdb_path: str, max_keys: int = 5) -> List[str]:
        """从__order__获取视频键"""
        env = self.open_lmdb_database(lmdb_path)
        video_keys = []
        
        with env.begin(write=False) as txn:
            try:
                order_data = msgpack.loads(txn.get(b'__order__'))
                if order_data:
                    video_keys = order_data[:max_keys]
                    print(f"✅ 从__order__获取到 {len(video_keys)} 个视频键")
                else:
                    print("❌ __order__数据为空")
            except Exception as e:
                print(f"❌ 无法从__order__获取键: {e}")
        
        env.close()
        return video_keys
    
    def save_images(self, images: List[Image.Image], video_key: str) -> List[str]:
        """保存提取的多帧图像"""
        saved_paths = []
        try:
            safe_key = video_key.replace('/', '_').replace('\\', '_')
            
            for i, image in enumerate(images):
                filename = f"{safe_key}_frame_{i:02d}.jpg"
                filepath = os.path.join(self.image_save_dir, filename)
                image.save(filepath, 'JPEG', quality=95)
                saved_paths.append(filepath)
            
            print(f"💾 已保存 {len(images)} 帧图像: {safe_key}")
            return saved_paths
            
        except Exception as e:
            print(f"❌ 图像保存失败: {e}")
            return []
    
    def process_with_clip_qwen(self, image_lmdb_path: str, res_lmdb_path: str, 
                              mv_lmdb_path: str, max_videos: int = 3):
        """
        使用CLIP+Qwen处理LMDB中的多模态视频数据
        
        Args:
            image_lmdb_path: 图像LMDB路径
            res_lmdb_path: 残差LMDB路径
            mv_lmdb_path: 运动向量LMDB路径
            max_videos: 最大处理视频数
        """
        print(f"🎯 === 使用CLIP+Qwen处理多模态LMDB视频数据 ===")
        
        # 初始化模型
        if not self.initialize_models():
            print("💥 模型初始化失败，无法继续")
            return []
        
        # 获取视频键
        video_keys = self.get_video_keys_from_order(image_lmdb_path, max_videos)
        if not video_keys:
            print("❌ 无法获取视频键")
            return []
        
        results = []
        
        for i, key in enumerate(video_keys):
            try:
                print(f"\n📝 处理 {i+1}/{len(video_keys)}: {key}")
                
                # 提取多模态数据
                images, residuals, motion_vectors = self.extract_multimodal_frames_from_lmdb(
                    image_lmdb_path, res_lmdb_path, mv_lmdb_path, key, self.frames_per_video
                )
                
                if not images:
                    print(f"❌ 无法提取视频帧: {key}")
                    continue
                
                # 保存图像
                saved_paths = self.save_images(images, key)
                
                # 使用CLIP+Qwen生成视频描述
                print("🤖 正在使用CLIP+Qwen生成视频描述...")
                description = self.generate_qwen_description_with_clip(images, residuals, motion_vectors)
                
                result = {
                    'video_key': key,
                    'total_frames_extracted': len(images),
                    'frame_sizes': [img.size for img in images],
                    'saved_image_paths': saved_paths,
                    'clip_qwen_description': description,
                    'status': 'success' if not description.startswith('❌') else 'failed'
                }
                results.append(result)
                
                print(f"✅ CLIP+Qwen视频描述生成完成")
                print(f"📝 描述预览: {description[:150]}...")
                
                # 显示GPU内存使用情况
                if torch.cuda.is_available() and self.gpu_ids:
                    for gpu_id in self.gpu_ids:
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        print(f"🎮 GPU {gpu_id} 内存使用: {memory_allocated:.2f}GB")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()
                result = {
                    'video_key': key,
                    'total_frames_extracted': 0,
                    'frame_sizes': [],
                    'saved_image_paths': [],
                    'clip_qwen_description': f"处理失败: {str(e)}",
                    'status': 'failed'
                }
                results.append(result)
        
        # 保存结果
        self.save_results(results)
        
        success_count = len([r for r in results if r['status'] == 'success'])
        print(f"\n🎉 CLIP+Qwen视频处理完成！成功: {success_count}/{len(results)}")
        
        return results
    
    def save_results(self, results: List[dict]):
        """保存处理结果"""
        output_file = "/home/stu_b/BIKE/hmdb_qwen/clip_qwen_video_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🤖 CLIP+QWEN LMDB 多模态视频描述生成报告\n")
            f.write("=" * 80 + "\n\n")
            
            success_count = len([r for r in results if r['status'] == 'success'])
            total_count = len(results)
            
            f.write(f"📊 处理统计:\n")
            f.write(f"  总视频数: {total_count}\n")
            f.write(f"  成功数量: {success_count}\n")
            f.write(f"  失败数量: {total_count - success_count}\n")
            f.write(f"  成功率: {success_count/total_count*100:.1f}%\n")
            f.write(f"  每视频采样帧数: {self.frames_per_video}\n")
            f.write(f"  CLIP配置: {self.clip_config}\n")
            f.write(f"  使用GPU: {self.gpu_ids}\n\n")
            f.write(f"📁 图像保存目录: {self.image_save_dir}\n\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"📁 视频 {i+1}: {result['video_key']}\n")
                f.write(f"📊 状态: {result['status']}\n")
                f.write(f"🎬 提取帧数: {result['total_frames_extracted']}\n")
                if result['frame_sizes']:
                    f.write(f"📏 帧尺寸: {result['frame_sizes']}\n")
                if result['saved_image_paths']:
                    f.write(f"💾 保存的图像数量: {len(result['saved_image_paths'])}\n")
                    f.write(f"💾 首个图像路径: {result['saved_image_paths'][0] if result['saved_image_paths'] else 'None'}\n")
                f.write(f"🤖 CLIP+Qwen描述:\n{result['clip_qwen_description']}\n")
                f.write("─" * 80 + "\n\n")
        
        print(f"💾 CLIP+Qwen处理结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CLIP+Qwen LMDB视频描述生成器')
    parser.add_argument('--image_lmdb', type=str, required=True, help='图像LMDB路径')
    parser.add_argument('--res_lmdb', type=str, required=True, help='残差LMDB路径')
    parser.add_argument('--mv_lmdb', type=str, required=True, help='运动向量LMDB路径')
    parser.add_argument('--max_videos', type=int, default=3, help='最大处理视频数')
    parser.add_argument('--frames_per_video', type=int, default=5, help='每个视频采样帧数')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPU ID列表')
    parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help='CLIP架构')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度 (ViT-B: 512, ViT-L: 768)')
    
    args = parser.parse_args()
    
    # CLIP配置
    clip_config = {
        'arch': args.clip_arch,
        'embed_dim': args.embed_dim,
        'residual_layers_to_use': [0, 11],
        'mvs_layers_to_use': [0, 11]
    }
    
    # 创建生成器
    generator = QwenCLIPLMDBGenerator(
        frames_per_video=args.frames_per_video,
        gpu_ids=args.gpu_ids,
        clip_config=clip_config
    )
    
    # 处理视频
    results = generator.process_with_clip_qwen(
        args.image_lmdb, 
        args.res_lmdb, 
        args.mv_lmdb, 
        args.max_videos
    )
    
    print("🎉 处理完成！")


if __name__ == "__main__":
    main()