import os
import lmdb
import msgpack
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
from typing import Union, List, Tuple
import time
import argparse

class QwenOnlyLMDBGenerator:
    def __init__(self, frames_per_video=5, gpu_ids=None):
        """
        专门使用Qwen模型的LMDB I-frame描述生成器
        
        Args:
            frames_per_video: 每个视频采样的帧数
            gpu_ids: 指定使用的GPU ID列表，如[0, 1]或单个GPU如[0]
        """
        # GPU设置
        self.gpu_ids = gpu_ids if gpu_ids is not None else []
        self.setup_gpu_environment()
        
        self.frames_per_video = frames_per_video  # 新增：每个视频采样帧数
        print(f"🚀 使用设备: {self.device}")
        print(f"🎮 GPU设置: {self.gpu_ids}")
        print(f"🎬 每个视频采样帧数: {self.frames_per_video}")
        
        # 初始化时不加载模型
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # 创建图像保存目录
        self.image_save_dir = "/home/stu_b/BIKE/hmdb_qwen"
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"📁 图像保存目录: {self.image_save_dir}")
        
        print("✅ Qwen专用LMDB I-frame描述生成器初始化完成")
    
    def setup_gpu_environment(self):
        """
        设置GPU环境
        """
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将使用CPU")
            self.device = torch.device("cpu")
            self.gpu_ids = []
            return
        
        total_gpus = torch.cuda.device_count()
        print(f"🎮 系统可用GPU数量: {total_gpus}")
        
        if not self.gpu_ids:
            # 如果没有指定GPU，使用第一个可用的
            self.gpu_ids = [0] if total_gpus > 0 else []
            print(f"🎮 未指定GPU，自动使用: GPU {self.gpu_ids}")
        else:
            # 验证指定的GPU是否可用
            valid_gpu_ids = []
            for gpu_id in self.gpu_ids:
                if 0 <= gpu_id < total_gpus:
                    valid_gpu_ids.append(gpu_id)
                    print(f"✅ GPU {gpu_id} 可用")
                else:
                    print(f"❌ GPU {gpu_id} 不可用 (系统只有 {total_gpus} 个GPU)")
            
            self.gpu_ids = valid_gpu_ids
        
        if self.gpu_ids:
            # 设置CUDA_VISIBLE_DEVICES环境变量
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
            print(f"🎮 设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # 设置主设备
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            
            # 显示GPU信息
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"🎮 GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ 没有可用的GPU，将使用CPU")
            self.device = torch.device("cpu")
    
    def setup_hf_mirror(self):
        """
        设置Hugging Face镜像源
        """
        print("🔧 配置Hugging Face镜像源...")
        
        # 设置环境变量
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/stu_b/.cache/huggingface'
        
        # 创建缓存目录
        cache_dir = '/home/stu_b/.cache/huggingface'
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"✅ 镜像源设置完成: {os.environ.get('HF_ENDPOINT')}")
        print(f"✅ 缓存目录: {cache_dir}")
    
    def load_qwen_model_with_retry(self, model_name="Qwen/Qwen2-VL-2B-Instruct", max_retries=3):
        """
        带重试机制的Qwen模型加载
        """
        print(f"🔄 开始加载Qwen模型: {model_name}")
        
        # 设置镜像源
        self.setup_hf_mirror()
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 尝试 {attempt + 1}/{max_retries}...")
                
                # 导入必要的库
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                print("📦 导入库成功")
                
                # 加载处理器
                print("🔧 加载处理器...")
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir='/home/stu_b/.cache/huggingface'
                )
                print("✅ 处理器加载成功")
                
                # 加载模型
                print("🔧 加载模型...")
                if torch.cuda.is_available() and self.gpu_ids:
                    if len(self.gpu_ids) > 1:
                        # 多GPU设置
                        print(f"🎮 使用多GPU加载: {self.gpu_ids}")
                        device_map = "auto"  # 让transformers自动分配
                    else:
                        # 单GPU设置
                        print(f"🎮 使用单GPU加载: GPU {self.gpu_ids[0]}")
                        device_map = {"": self.gpu_ids[0]}
                    
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        trust_remote_code=True,
                        cache_dir='/home/stu_b/.cache/huggingface'
                    )
                else:
                    # CPU模式
                    print("🔧 使用CPU加载模型...")
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        cache_dir='/home/stu_b/.cache/huggingface'
                    )
                    self.model.to(self.device)
                
                self.model.eval()
                self.model_loaded = True
                
                # 显示模型加载后的GPU内存使用情况
                if torch.cuda.is_available() and self.gpu_ids:
                    for gpu_id in self.gpu_ids:
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        print(f"🎮 GPU {gpu_id} 内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
                
                print("🎉 Qwen模型加载成功！")
                return True
                
            except Exception as e:
                print(f"❌ 尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    print("⏳ 等待5秒后重试...")
                    time.sleep(5)
                else:
                    print("💥 所有尝试都失败了")
                    return False
        
        return False
    
    def check_local_qwen_model(self):
        """
        检查本地是否已有Qwen模型
        """
        possible_paths = [
            '/home/stu_b/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct',
            '/home/stu_b/.cache/huggingface/transformers',
            '~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct'
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                print(f"✅ 找到本地模型: {expanded_path}")
                return expanded_path
        
        print("🔍 未找到本地Qwen模型")
        return None
    
    def load_local_qwen_model(self, local_path):
        """
        加载本地Qwen模型
        """
        try:
            print(f"🔧 从本地路径加载Qwen模型: {local_path}")
            
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                local_path,
                trust_remote_code=True,
                local_files_only=True
            )
            print("✅ 本地处理器加载成功")
            
            # 加载模型
            if torch.cuda.is_available() and self.gpu_ids:
                if len(self.gpu_ids) > 1:
                    device_map = "auto"
                else:
                    device_map = {"": self.gpu_ids[0]}
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    local_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                )
                self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            print("🎉 本地Qwen模型加载成功！")
            return True
            
        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            return False
    
    def initialize_qwen_model(self):
        """
        初始化Qwen模型（优先本地，然后在线）
        """
        print("🎯 === 初始化Qwen模型 ===")
        
        # 首先检查本地模型
        local_path = self.check_local_qwen_model()
        if local_path:
            if self.load_local_qwen_model(local_path):
                return True
        
        # 如果本地没有，尝试在线下载
        print("🌐 本地模型不可用，尝试在线下载...")
        return self.load_qwen_model_with_retry()
    
    def save_images(self, images: List[Image.Image], video_key: str) -> List[str]:
        """
        保存提取的多帧图像
        
        Args:
            images: 图像列表
            video_key: 视频键
        
        Returns:
            保存路径列表
        """
        saved_paths = []
        try:
            # 创建安全的文件名
            safe_key = video_key.replace('/', '_').replace('\\', '_')
            
            for i, image in enumerate(images):
                filename = f"{safe_key}_frame_{i:02d}.jpg"
                filepath = os.path.join(self.image_save_dir, filename)
                
                # 保存图像
                image.save(filepath, 'JPEG', quality=95)
                saved_paths.append(filepath)
            
            print(f"💾 已保存 {len(images)} 帧图像: {safe_key}")
            return saved_paths
            
        except Exception as e:
            print(f"❌ 图像保存失败: {e}")
            return []
    
    def resize_image_compatible(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """
        兼容不同PIL版本的图像缩放方法
        """
        try:
            # 尝试新版本PIL的方法
            if hasattr(Image, 'Resampling'):
                return image.resize(target_size, Image.Resampling.LANCZOS)
            else:
                # 旧版本PIL的方法
                return image.resize(target_size, Image.LANCZOS)
        except Exception as e:
            print(f"🔧 图像缩放失败，使用默认方法: {e}")
            return image.resize(target_size)
    
    def generate_qwen_video_description(self, images: List[Image.Image], 
                                      prompt="Analyze these 5 sequential video frames and identify the primary human action being performed. Provide:\n- Action: [specific action name]\n- Category: [action type/category]\n- Description: [brief movement description]\nFocus on the main action, not background details.") -> str:
        """
        使用多帧图像生成视频描述
        
        Args:
            images: 图像列表（按时间顺序）
            prompt: 提示词
        
        Returns:
            生成的视频描述
        """
        print(f"🤖 正在分析 {len(images)} 帧图像生成视频描述...")
        
        try:
            # 调整所有图像尺寸
            target_size = (448, 448)
            resized_images = []
            for i, image in enumerate(images):
                resized_img = self.resize_image_compatible(image, target_size)
                resized_images.append(resized_img)
                print(f"🔧 第{i+1}帧已调整到: {resized_img.size}")
            
            # 构造多图像的prompt
            # 为每个图像添加标识
            image_tokens = ""
            for i in range(len(resized_images)):
                image_tokens += f"<|vision_start|><|image_pad|><|vision_end|>"
            
            # 构造完整的prompt
            full_prompt = f"These are {len(images)} sequential frames from a video in chronological order. {prompt}"
            text_prompt = f"<|im_start|>user\n{image_tokens}{full_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            print(f"🔧 构造的prompt长度: {len(text_prompt)}")
            
            # 使用processor处理多图像
            inputs = self.processor(
                text=[text_prompt],
                images=resized_images,  # 传入图像列表
                return_tensors="pt",
                padding=True
            )
            
            print(f"🔧 多帧处理成功，输入键: {list(inputs.keys())}")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"🔧 {key} shape: {value.shape}")
            
            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,  # 使用你推荐的150 tokens
                    do_sample=True,
                    temperature=0.1,     # 使用你推荐的低温度
                    top_p=0.8,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 解码
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_length:]
            
            output_text = self.processor.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True
            )
            
            return output_text if output_text.strip() else "生成的视频描述为空"
            
        except Exception as e:
            error_msg = f"❌ 视频描述生成失败: {str(e)}"
            print(error_msg)
            return error_msg
    
    def uniform_sample_frames(self, total_frames: int, num_samples: int) -> List[int]:
        """
        均匀采样帧索引
        
        Args:
            total_frames: 总帧数
            num_samples: 采样数量
        
        Returns:
            采样的帧索引列表
        """
        if total_frames <= num_samples:
            # 如果总帧数不够，就返回所有帧的索引
            return list(range(total_frames))
        
        # 均匀采样
        indices = []
        step = total_frames / num_samples
        for i in range(num_samples):
            idx = int(i * step)
            # 确保不超出范围
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        
        # 去重并排序
        indices = sorted(list(set(indices)))
        print(f"🎯 从 {total_frames} 帧中均匀采样 {len(indices)} 帧: {indices}")
        return indices

    def open_lmdb_database(self, lmdb_path: str):
        """
        打开LMDB数据库
        """
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB数据库不存在: {lmdb_path}")
        
        env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                       readonly=True, lock=False,
                       readahead=False, meminit=False)
        
        print(f"📂 成功打开LMDB数据库: {lmdb_path}")
        return env
    
    def get_video_keys_from_order(self, lmdb_path: str, max_keys: int = 5) -> List[str]:
        """
        从__order__获取视频键
        """
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
    
    def extract_multiple_frames_from_lmdb(self, lmdb_path: str, order_key: str, num_frames: int = None) -> List[Image.Image]:
        """
        从LMDB提取多帧图像（均匀采样）
        
        Args:
            lmdb_path: LMDB路径
            order_key: 视频键
            num_frames: 采样帧数，如果为None则使用self.frames_per_video
        
        Returns:
            图像列表
        """
        if num_frames is None:
            num_frames = self.frames_per_video
            
        env = self.open_lmdb_database(lmdb_path)
        images = []
        
        with env.begin(write=False) as txn:
            try:
                order_data = msgpack.loads(txn.get(b'__order__'))
                keys_data = msgpack.loads(txn.get(b'__keys__'))
                
                if order_key in order_data:
                    order_index = order_data.index(order_key)
                    actual_key = keys_data[order_index]
                    video_data = txn.get(actual_key)
                else:
                    video_data = txn.get(order_key.encode() if isinstance(order_key, str) else order_key)
                    
            except Exception as e:
                video_data = txn.get(order_key.encode() if isinstance(order_key, str) else order_key)
            
            if video_data is None:
                env.close()
                raise ValueError(f"未找到视频数据: {order_key}")
            
            # 解码视频数据
            frames_data = msgpack.loads(video_data, raw=True)
            total_frames = len(frames_data)
            
            print(f"📊 视频 {order_key} 总帧数: {total_frames}")
            
            # 均匀采样帧索引
            frame_indices = self.uniform_sample_frames(total_frames, num_frames)
            
            # 提取对应帧
            for i, frame_idx in enumerate(frame_indices):
                try:
                    frame_bytes = frames_data[frame_idx]
                    image = Image.open(BytesIO(frame_bytes)).convert('RGB')
                    images.append(image)
                    print(f"✅ 提取第 {i+1}/{len(frame_indices)} 帧 (索引{frame_idx}): {image.size}")
                except Exception as e:
                    print(f"❌ 提取帧 {frame_idx} 失败: {e}")
            
            print(f"🎬 成功提取 {len(images)}/{len(frame_indices)} 帧")
        
        env.close()
        return images
    
    def process_with_qwen(self, lmdb_path: str, max_videos: int = 3):
        """
        使用Qwen模型处理LMDB中的视频帧（多帧模式）
        """
        print(f"🎯 === 使用Qwen处理LMDB视频帧（每视频{self.frames_per_video}帧）===")
        
        # 初始化Qwen模型
        if not self.initialize_qwen_model():
            print("💥 Qwen模型初始化失败，无法继续")
            return []
        
        # 获取视频键
        video_keys = self.get_video_keys_from_order(lmdb_path, max_videos)
        if not video_keys:
            print("❌ 无法获取视频键")
            return []
        
        results = []
        
        for i, key in enumerate(video_keys):
            try:
                print(f"\n📝 处理 {i+1}/{len(video_keys)}: {key}")
                
                # 提取多帧图像
                images = self.extract_multiple_frames_from_lmdb(lmdb_path, key, self.frames_per_video)
                
                if not images:
                    print(f"❌ 无法提取视频帧: {key}")
                    continue
                
                # 保存图像
                saved_paths = self.save_images(images, key)
                
                # 使用Qwen生成视频描述
                print("🤖 正在使用Qwen生成视频描述...")
                description = self.generate_qwen_video_description(images)
                
                result = {
                    'video_key': key,
                    'total_frames_extracted': len(images),
                    'frame_sizes': [img.size for img in images],
                    'saved_image_paths': saved_paths,
                    'qwen_video_description': description,
                    'status': 'success' if not description.startswith('❌') else 'failed'
                }
                results.append(result)
                
                print(f"✅ Qwen视频描述生成完成")
                print(f"📝 描述预览: {description[:150]}...")
                
                # 显示GPU内存使用情况
                if torch.cuda.is_available() and self.gpu_ids:
                    for gpu_id in self.gpu_ids:
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        print(f"🎮 GPU {gpu_id} 内存使用: {memory_allocated:.2f}GB")
                
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                result = {
                    'video_key': key,
                    'total_frames_extracted': 0,
                    'frame_sizes': [],
                    'saved_image_paths': [],
                    'qwen_video_description': f"处理失败: {str(e)}",
                    'status': 'failed'
                }
                results.append(result)
        
        # 保存结果
        self.save_qwen_results(results)
        
        success_count = len([r for r in results if r['status'] == 'success'])
        print(f"\n🎉 Qwen视频处理完成！成功: {success_count}/{len(results)}")
        
        return results
    
    def save_qwen_results(self, results: List[dict]):
        """
        保存Qwen处理结果
        """
        output_file = "/home/stu_b/BIKE/hmdb_qwen/qwen_video_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🤖 QWEN LMDB 多帧视频描述生成报告\n")
            f.write("=" * 80 + "\n\n")
            
            success_count = len([r for r in results if r['status'] == 'success'])
            total_count = len(results)
            
            f.write(f"📊 处理统计:\n")
            f.write(f"  总视频数: {total_count}\n")
            f.write(f"  成功数量: {success_count}\n")
            f.write(f"  失败数量: {total_count - success_count}\n")
            f.write(f"  成功率: {success_count/total_count*100:.1f}%\n")
            f.write(f"  每视频采样帧数: {self.frames_per_video}\n")
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
                f.write(f"🤖 Qwen视频描述:\n{result['qwen_video_description']}\n")
                f.write("─" * 80 + "\n\n")
        
        print(f"💾 Qwen视频处理结果已保存到: {output_file}")


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Qwen专用LMDB多帧视频描述生成器')
    
    parser.add_argument('--gpus', type=str, default='0', 
                       help='指定使用的GPU ID，用逗号分隔，如: 0 或 0,1,2 (默认: 0)')
    
    parser.add_argument('--frames', type=int, default=5,
                       help='每个视频采样的帧数 (默认: 5)')
    
    parser.add_argument('--max_videos', type=int, default=10,
                       help='处理的最大视频数量 (默认: 10)')
    
    parser.add_argument('--lmdb_path', type=str, 
                       default='/mnt/data/hmdb51/HMDB51_lmdb/hmdb51_compressed_frames.lmdb',
                       help='LMDB数据库路径')
    
    parser.add_argument('--output_dir', type=str, 
                       default='/home/stu_b/BIKE/hmdb_qwen',
                       help='输出目录路径 (默认: /home/stu_b/BIKE/hmdb_qwen)')
    
    parser.add_argument('--model_name', type=str, 
                       default='Qwen/Qwen2-VL-2B-Instruct',
                       help='Qwen模型名称 (默认: Qwen/Qwen2-VL-2B-Instruct)')
    
    return parser.parse_args()


def main():
    """
    主函数 - 专门使用Qwen模型处理多帧视频
    """
    print("🤖 === Qwen专用LMDB多帧视频描述生成器 ===")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 解析GPU参数
    if args.gpus.lower() == 'cpu':
        gpu_ids = []
        print("🔧 使用CPU模式")
    else:
        try:
            gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
            print(f"🎮 指定GPU: {gpu_ids}")
        except ValueError:
            print(f"❌ GPU参数格式错误: {args.gpus}")
            print("💡 正确格式: --gpus 0 或 --gpus 0,1,2 或 --gpus cpu")
            return
    
    # 显示配置信息
    print(f"📋 运行配置:")
    print(f"  🎮 GPU设置: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"  🎬 每视频帧数: {args.frames}")
    print(f"  📊 最大视频数: {args.max_videos}")
    print(f"  📂 LMDB路径: {args.lmdb_path}")
    print(f"  📁 输出目录: {args.output_dir}")
    print(f"  🤖 模型名称: {args.model_name}")
    
    # 验证LMDB路径
    if not os.path.exists(args.lmdb_path):
        print(f"❌ LMDB路径不存在: {args.lmdb_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 初始化生成器
        generator = QwenOnlyLMDBGenerator(
            frames_per_video=args.frames,
            gpu_ids=gpu_ids
        )
        
        # 设置输出目录
        generator.image_save_dir = args.output_dir
        
        # 使用Qwen处理多帧视频
        results = generator.process_with_qwen(args.lmdb_path, max_videos=args.max_videos)
        
        if results:
            success_count = len([r for r in results if r['status'] == 'success'])
            total_frames = sum([r['total_frames_extracted'] for r in results])
            print(f"\n🎉 处理完成！")
            print(f"📊 成功处理 {success_count} 个视频")
            print(f"🎬 总共提取 {total_frames} 帧图像")
            print(f"📁 图像保存在: {generator.image_save_dir}")
            
            # 显示最终GPU内存使用情况
            if torch.cuda.is_available() and gpu_ids:
                print(f"\n🎮 最终GPU内存使用情况:")
                for gpu_id in gpu_ids:
                    memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    print(f"  GPU {gpu_id}: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        else:
            print("\n💥 处理失败，请检查网络连接和模型配置")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n💥 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
