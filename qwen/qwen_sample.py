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

def pil_from_raw_rgb(raw):
    """从原始RGB数据创建PIL图像"""
    return Image.open(BytesIO(raw)).convert('RGB')

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
        
        self.frames_per_video = frames_per_video
        print(f"🚀 初始化完成 - 设备: {self.device}, GPU: {self.gpu_ids}, 每视频帧数: {self.frames_per_video}")
        
        # 初始化时不加载模型
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # 初始化LMDB相关变量
        self.env_i = None
        self.db_length = None
        self.db_keys = None
        self.db_order = None
        self.vlen_list = None
        self.get_video_id = None
    
    def setup_gpu_environment(self):
        """设置GPU环境"""
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.gpu_ids = []
            return
        
        total_gpus = torch.cuda.device_count()
        
        if not self.gpu_ids:
            self.gpu_ids = [0] if total_gpus > 0 else []
        else:
            # 验证指定的GPU是否可用
            valid_gpu_ids = []
            for gpu_id in self.gpu_ids:
                if 0 <= gpu_id < total_gpus:
                    valid_gpu_ids.append(gpu_id)
            self.gpu_ids = valid_gpu_ids
        
        if self.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        else:
            self.device = torch.device("cpu")
    
    def setup_hf_mirror(self):
        """设置Hugging Face镜像源"""
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/stu_b/.cache/huggingface'
        cache_dir = '/home/stu_b/.cache/huggingface'
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_qwen_model_with_retry(self, model_name="Qwen/Qwen2-VL-2B-Instruct", max_retries=3):
        """带重试机制的Qwen模型加载"""
        print(f"🔄 加载Qwen模型: {model_name}")
        self.setup_hf_mirror()
        
        for attempt in range(max_retries):
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                # 加载处理器
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir='/home/stu_b/.cache/huggingface'
                )
                
                # 加载模型
                if torch.cuda.is_available() and self.gpu_ids:
                    if len(self.gpu_ids) > 1:
                        device_map = "auto"
                    else:
                        device_map = {"": self.gpu_ids[0]}
                    
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        trust_remote_code=True,
                        cache_dir='/home/stu_b/.cache/huggingface'
                    )
                else:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        cache_dir='/home/stu_b/.cache/huggingface'
                    )
                    self.model.to(self.device)
                
                self.model.eval()
                self.model_loaded = True
                print("✅ Qwen模型加载成功")
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"❌ 尝试 {attempt + 1} 失败，重试中...")
                    time.sleep(5)
                else:
                    print(f"💥 模型加载失败: {e}")
                    return False
        
        return False
    
    def check_local_qwen_model(self):
        """检查本地是否已有Qwen模型"""
        possible_paths = [
            '/home/stu_b/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct',
            '/home/stu_b/.cache/huggingface/transformers',
            '~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct'
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        return None
    
    def load_local_qwen_model(self, local_path):
        """加载本地Qwen模型"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(
                local_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
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
            print("✅ 本地Qwen模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            return False
    
    def initialize_qwen_model(self):
        """初始化Qwen模型（优先本地，然后在线）"""
        # 首先检查本地模型
        local_path = self.check_local_qwen_model()
        if local_path:
            if self.load_local_qwen_model(local_path):
                return True
        
        # 如果本地没有，尝试在线下载
        return self.load_qwen_model_with_retry()
    
    def read_txt_file(self, txt_path: str) -> list:
        """
        读取txt文件获取视频键值列表
        
        Args:
            txt_path: txt文件路径
        
        Returns:
            视频键值列表
        """
        video_keys = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 分割每行，取第一列作为键值
                        parts = line.split()
                        if parts:
                            video_key = parts[0]
                            video_keys.append(video_key)
            
            print(f"📊 从 {txt_path} 读取到 {len(video_keys)} 个视频键")
            return video_keys
            
        except Exception as e:
            print(f"❌ 读取txt文件失败: {e}")
            return []
    
    def initialize_lmdb(self, lmdb_path: str):
        """
        初始化LMDB数据库，加载元数据
        
        Args:
            lmdb_path: LMDB数据库路径
        """
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB数据库不存在: {lmdb_path}")
        
        print("🔗 初始化LMDB数据库...")
        
        # 打开LMDB环境
        self.env_i = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                              readonly=True, lock=False,
                              readahead=False, meminit=False)
        
        # 加载元数据
        with self.env_i.begin(write=False) as txn:
            try:
                self.db_length = msgpack.loads(txn.get(b'__len__'))
                self.db_keys = msgpack.loads(txn.get(b'__keys__'))
                self.db_order = msgpack.loads(txn.get(b'__order__'))
                self.vlen_list = msgpack.loads(txn.get(b'__vlen__'))
                
                print(f"✅ 成功加载LMDB元数据:")
                print(f"  - 数据库长度: {self.db_length}")
                print(f"  - 键值数量: {len(self.db_keys) if self.db_keys else 0}")
                print(f"  - 顺序数量: {len(self.db_order) if self.db_order else 0}")
                print(f"  - 长度列表: {len(self.vlen_list) if self.vlen_list else 0}")
                
            except Exception as e:
                print(f"❌ 加载LMDB元数据失败: {e}")
                raise
        
        # 创建视频ID映射
        if self.db_order:
            self.get_video_id = dict(zip([i for i in self.db_order],
                                       ['%09d' % i for i in range(len(self.db_order))]))
            print(f"✅ 创建视频ID映射，共 {len(self.get_video_id)} 个条目")
            
            # 显示前几个映射示例
            sample_items = list(self.get_video_id.items())[:5]
            print("📋 映射示例:")
            for vname, vid in sample_items:
                print(f"  {vname} -> {vid}")
        else:
            raise ValueError("无法获取db_order数据")
    
    def extract_multiple_frames_from_lmdb(self, vname: str, num_frames: int = None) -> List[Image.Image]:
        """
        使用参考代码的逻辑从LMDB提取多帧图像（均匀采样）
        
        Args:
            vname: 视频名称（来自txt文件第一列）
            num_frames: 要提取的帧数
        
        Returns:
            图像列表
        """
        if num_frames is None:
            num_frames = self.frames_per_video
            
        try:
            # 检查视频名称是否在映射中
            if vname not in self.get_video_id:
                print(f"⚠️ 视频 {vname} 不在映射中")
                return []
            
            # 获取视频ID
            video_id = self.get_video_id[vname]
            
            # 从LMDB读取原始数据
            with self.env_i.begin(write=False) as txn:
                raw_i = msgpack.loads(txn.get(video_id.encode('ascii')), 
                                    raw=True, strict_map_key=False)
            
            if not raw_i:
                print(f"⚠️ 无法获取视频 {vname} 的数据")
                return []
            
            total_frames = len(raw_i)
            
            # 均匀采样帧索引
            frame_indices = self.uniform_sample_frames(total_frames, num_frames)
            
            # 提取对应帧
            images = []
            for frame_idx in frame_indices:
                try:
                    frame_raw = raw_i[frame_idx]
                    # 使用参考代码的函数转换为PIL图像
                    image = pil_from_raw_rgb(frame_raw)
                    images.append(image)
                except Exception as e:
                    print(f"❌ 提取帧 {frame_idx} 失败: {e}")
            
            return images
            
        except Exception as e:
            print(f"❌ 提取多帧失败 {vname}: {e}")
            return []
    
    def resize_image_compatible(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """兼容不同PIL版本的图像缩放方法"""
        try:
            if hasattr(Image, 'Resampling'):
                return image.resize(target_size, Image.Resampling.LANCZOS)
            else:
                return image.resize(target_size, Image.LANCZOS)
        except Exception as e:
            return image.resize(target_size)
    
    def generate_qwen_video_description(self, images: List[Image.Image], 
                                      prompt="What is the action category? Provide the phrases of the five most likely categories") -> str:
        """使用多帧图像生成视频描述"""
        try:
            # 调整所有图像尺寸
            target_size = (448, 448)
            resized_images = [self.resize_image_compatible(img, target_size) for img in images]
            
            # 构造多图像的prompt
            image_tokens = ""
            for i in range(len(resized_images)):
                image_tokens += f"<|vision_start|><|image_pad|><|vision_end|>"
            
            full_prompt = f"These are {len(images)} sequential frames from a video in chronological order. {prompt}"
            text_prompt = f"<|im_start|>user\n{image_tokens}{full_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 使用processor处理多图像
            inputs = self.processor(
                text=[text_prompt],
                images=resized_images,
                return_tensors="pt",
                padding=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.1,
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
            return f"❌ 视频描述生成失败: {str(e)}"
    
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
        
        return sorted(list(set(indices)))
    
    def process_with_qwen_from_txt(self, txt_path: str, lmdb_path: str, output_dir: str, max_videos: int = 3):
        """
        使用txt文件和LMDB处理视频（不保存图像）
        
        Args:
            txt_path: txt文件路径
            lmdb_path: LMDB数据库路径
            output_dir: 输出目录（仅用于保存结果文本）
            max_videos: 最大处理视频数
        """
        print(f"🎯 开始处理 {max_videos} 个视频，每视频 {self.frames_per_video} 帧")
        
        # 初始化Qwen模型
        if not self.initialize_qwen_model():
            print("💥 Qwen模型初始化失败")
            return []
        
        # 读取txt文件获取视频键
        video_keys = self.read_txt_file(txt_path)
        if not video_keys:
            print("❌ 无法获取视频键")
            return []
        
        # 初始化LMDB
        try:
            self.initialize_lmdb(lmdb_path)
        except Exception as e:
            print(f"❌ LMDB初始化失败: {e}")
            return []
        
        # 限制处理数量
        if max_videos and max_videos < len(video_keys):
            video_keys = video_keys[:max_videos]
            print(f"📊 限制处理数量为: {max_videos}")
        
        results = []
        success_count = 0
        
        try:
            for i, key in enumerate(video_keys):
                try:
                    print(f"📝 处理 {i+1}/{len(video_keys)}: {key}")
                    
                    # 提取多帧图像
                    images = self.extract_multiple_frames_from_lmdb(key, self.frames_per_video)
                    
                    if not images:
                        print(f"❌ 无法提取视频帧: {key}")
                        result = {
                            'video_key': key,
                            'total_frames_extracted': 0,
                            'qwen_video_description': "无法提取视频帧",
                            'status': 'failed'
                        }
                        results.append(result)
                        continue
                    
                    # 使用Qwen生成视频描述（不保存图像）
                    description = self.generate_qwen_video_description(images)
                    
                    if not description.startswith('❌'):
                        success_count += 1
                    
                    result = {
                        'video_key': key,
                        'total_frames_extracted': len(images),
                        'qwen_video_description': description,
                        'status': 'success' if not description.startswith('❌') else 'failed'
                    }
                    results.append(result)
                    
                    # 简化输出 - 只显示描述的前100个字符
                    print(f"✅ 完成 - 描述: {description[:100]}...")
                    
                except Exception as e:
                    print(f"❌ 处理失败: {e}")
                    result = {
                        'video_key': key,
                        'total_frames_extracted': 0,
                        'qwen_video_description': f"处理失败: {str(e)}",
                        'status': 'failed'
                    }
                    results.append(result)
        
        finally:
            # 确保关闭数据库连接
            if self.env_i:
                self.env_i.close()
                print("🔒 LMDB数据库连接已关闭")
        
        # 保存结果到输出目录
        self.save_qwen_results(results, output_dir)
        
        print(f"🎉 处理完成！成功: {success_count}/{len(results)}")
        
        return results
    
    def save_qwen_results(self, results: List[dict], output_dir: str):
        """保存Qwen处理结果（不包含图像信息）"""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "qwen_video_results.txt")
        
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
            
            # LMDB信息
            if self.db_length:
                f.write(f"📂 LMDB信息:\n")
                f.write(f"  数据库长度: {self.db_length}\n")
                f.write(f"  键值数量: {len(self.db_keys) if self.db_keys else 0}\n")
                f.write(f"  顺序数量: {len(self.db_order) if self.db_order else 0}\n")
                f.write(f"  映射数量: {len(self.get_video_id) if self.get_video_id else 0}\n\n")
            
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"📁 视频 {i+1}: {result['video_key']}\n")
                f.write(f"📊 状态: {result['status']}\n")
                f.write(f"🎬 提取帧数: {result['total_frames_extracted']}\n")
                f.write(f"🤖 Qwen视频描述:\n{result['qwen_video_description']}\n")
                f.write("─" * 80 + "\n\n")
        
        print(f"💾 结果已保存到: {output_file}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Qwen专用LMDB多帧视频描述生成器（无图像保存）')
    
    parser.add_argument('--txt_path', type=str, 
                       default='/mnt/data/hmdb51/HMDB51_lmdb/annotation_file/hmdb51_test.txt',
                       help='txt文件路径')
    
    parser.add_argument('--lmdb_path', type=str, 
                       default='/mnt/data/hmdb51/HMDB51_lmdb/hmdb51_compressed_frames.lmdb',
                       help='LMDB数据库路径')
    
    parser.add_argument('--output_dir', type=str, 
                       default='/home/stu_b/BIKE/hmdb_qwen_test',
                       help='输出目录路径（仅保存文本结果）')
    
    parser.add_argument('--gpus', type=str, default='0', 
                       help='指定使用的GPU ID，用逗号分隔 (默认: 0)')
    
    parser.add_argument('--frames', type=int, default=5,
                       help='每个视频采样的帧数 (默认: 5)')
    
    parser.add_argument('--max_videos', type=int, default=3000,
                       help='处理的最大视频数量 (默认: 3000)')
    
    return parser.parse_args()


def main():
    """主函数 - 专门使用Qwen模型处理多帧视频（不保存图像）"""
    print("🤖 === Qwen专用LMDB多帧视频描述生成器（无图像保存版本）===")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 解析GPU参数
    if args.gpus.lower() == 'cpu':
        gpu_ids = []
    else:
        try:
            gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
        except ValueError:
            print(f"❌ GPU参数格式错误: {args.gpus}")
            return
    
    # 验证文件路径
    if not os.path.exists(args.txt_path):
        print(f"❌ txt文件不存在: {args.txt_path}")
        return
    
    if not os.path.exists(args.lmdb_path):
        print(f"❌ LMDB路径不存在: {args.lmdb_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 显示配置信息
    print(f"📋 运行配置:")
    print(f"  📄 txt文件: {args.txt_path}")
    print(f"  📂 LMDB路径: {args.lmdb_path}")
    print(f"  📁 输出目录: {args.output_dir}")
    print(f"  🎮 GPU设置: {args.gpus}")
    print(f"  🎬 每视频帧数: {args.frames}")
    print(f"  📊 最大处理数: {args.max_videos}")
    print(f"  🖼️ 图像保存: 禁用")
    
    try:
        # 初始化生成器
        generator = QwenOnlyLMDBGenerator(
            frames_per_video=args.frames,
            gpu_ids=gpu_ids
        )
        
        # 使用txt文件和LMDB处理多帧视频（不保存图像）
        results = generator.process_with_qwen_from_txt(
            txt_path=args.txt_path,
            lmdb_path=args.lmdb_path,
            output_dir=args.output_dir,
            max_videos=args.max_videos
        )
        
        if results:
            success_count = len([r for r in results if r['status'] == 'success'])
            print(f"📊 最终统计: 成功 {success_count} 个视频")
        else:
            print("💥 处理失败")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n💥 程序执行失败: {e}")


if __name__ == "__main__":
    main()
