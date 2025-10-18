import os
import lmdb
import msgpack
from io import BytesIO
from PIL import Image
import argparse

def pil_from_raw_rgb(raw):
    """从原始RGB数据创建PIL图像"""
    return Image.open(BytesIO(raw)).convert('RGB')

class SimpleLMDBImageExtractor:
    def __init__(self, output_dir="/home/stu_b/BIKE/hmdb_qwen_1"):
        """
        简单的LMDB图像提取器
        
        Args:
            output_dir: 图像保存目录
        """
        self.image_save_dir = output_dir
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"🚀 初始化完成 - 图像保存目录: {self.image_save_dir}")
        
        # 初始化LMDB相关变量
        self.env_i = None
        self.db_length = None
        self.db_keys = None
        self.db_order = None
        self.vlen_list = None
        self.get_video_id = None
    
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
    
    def extract_first_frame(self, vname: str) -> Image.Image:
        """
        使用参考代码的逻辑从LMDB提取第一帧图像
        
        Args:
            vname: 视频名称（来自txt文件第一列）
        
        Returns:
            第一帧图像，如果失败返回None
        """
        try:
            # 检查视频名称是否在映射中
            if vname not in self.get_video_id:
                print(f"⚠️ 视频 {vname} 不在映射中")
                return None
            
            # 获取视频ID
            video_id = self.get_video_id[vname]
            
            # 从LMDB读取原始数据
            with self.env_i.begin(write=False) as txn:
                raw_i = msgpack.loads(txn.get(video_id.encode('ascii')), 
                                    raw=True, strict_map_key=False)
            
            if not raw_i:
                print(f"⚠️ 无法获取视频 {vname} 的数据")
                return None
            
            # 提取第一帧 (索引0)
            first_frame_raw = raw_i[0]
            
            # 使用参考代码的函数转换为PIL图像
            image = pil_from_raw_rgb(first_frame_raw)
            
            return image
            
        except Exception as e:
            print(f"❌ 提取帧失败 {vname}: {e}")
            return None
    
    def save_image(self, image: Image.Image, video_key: str) -> str:
        """
        保存图像
        
        Args:
            image: 图像
            video_key: 视频键（用作文件名）
        
        Returns:
            保存路径
        """
        try:
            # 创建安全的文件名
            safe_key = video_key.replace('/', '_').replace('\\', '_').replace('(', '').replace(')', '').replace(' ', '_')
            filename = f"{safe_key}.jpg"
            filepath = os.path.join(self.image_save_dir, filename)
            
            # 保存图像
            image.save(filepath, 'JPEG', quality=95)
            return filepath
            
        except Exception as e:
            print(f"❌ 图像保存失败: {e}")
            return ""
    
    def process_videos(self, txt_path: str, lmdb_path: str, max_videos: int = None):
        """
        处理视频：读取txt文件，从LMDB提取第一帧并保存
        
        Args:
            txt_path: txt文件路径
            lmdb_path: LMDB数据库路径
            max_videos: 最大处理视频数，None表示处理所有
        """
        print(f"🎯 开始处理视频")
        
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
        not_found_count = 0
        
        try:
            # 处理每个视频
            for i, video_key in enumerate(video_keys):
                try:
                    if i % 50 == 0:  # 每50个显示一次进度
                        print(f"📝 处理进度: {i+1}/{len(video_keys)} ({(i+1)/len(video_keys)*100:.1f}%)")
                    
                    # 从LMDB提取第一帧
                    image = self.extract_first_frame(video_key)
                    
                    if image is None:
                        not_found_count += 1
                        result = {
                            'video_key': video_key,
                            'saved_image_path': "",
                            'status': 'failed',
                            'error': '无法提取图像或视频不存在'
                        }
                        results.append(result)
                        continue
                    
                    # 保存图像
                    saved_path = self.save_image(image, video_key)
                    
                    if saved_path:
                        success_count += 1
                        if success_count % 100 == 0:  # 每100个成功显示一次
                            print(f"✅ 已成功处理: {success_count} 个视频")
                        
                        result = {
                            'video_key': video_key,
                            'saved_image_path': saved_path,
                            'image_size': image.size,
                            'status': 'success'
                        }
                    else:
                        result = {
                            'video_key': video_key,
                            'saved_image_path': "",
                            'status': 'failed',
                            'error': '保存失败'
                        }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"❌ 处理失败 {video_key}: {e}")
                    result = {
                        'video_key': video_key,
                        'saved_image_path': "",
                        'status': 'failed',
                        'error': str(e)
                    }
                    results.append(result)
        
        finally:
            # 确保关闭数据库连接
            if self.env_i:
                self.env_i.close()
                print("🔒 LMDB数据库连接已关闭")
        
        # 保存处理结果
        self.save_results(results, txt_path)
        
        print(f"🎉 处理完成！")
        print(f"  ✅ 成功: {success_count}/{len(video_keys)} ({success_count/len(video_keys)*100:.1f}%)")
        print(f"  ❌ 失败: {len(video_keys) - success_count}")
        print(f"  🔍 未找到: {not_found_count}")
        
        return results
    
    def save_results(self, results: list, txt_path: str):
        """保存处理结果"""
        output_file = os.path.join(self.image_save_dir, "extraction_results.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("📊 LMDB图像提取结果报告 (使用参考代码逻辑)\n")
            f.write("=" * 80 + "\n\n")
            
            success_count = len([r for r in results if r['status'] == 'success'])
            total_count = len(results)
            
            f.write(f"📋 处理统计:\n")
            f.write(f"  源txt文件: {txt_path}\n")
            f.write(f"  总视频数: {total_count}\n")
            f.write(f"  成功数量: {success_count}\n")
            f.write(f"  失败数量: {total_count - success_count}\n")
            f.write(f"  成功率: {success_count/total_count*100:.1f}%\n")
            f.write(f"  图像保存目录: {self.image_save_dir}\n\n")
            
            # LMDB信息
            if self.db_length:
                f.write(f"📂 LMDB信息:\n")
                f.write(f"  数据库长度: {self.db_length}\n")
                f.write(f"  键值数量: {len(self.db_keys) if self.db_keys else 0}\n")
                f.write(f"  顺序数量: {len(self.db_order) if self.db_order else 0}\n")
                f.write(f"  映射数量: {len(self.get_video_id) if self.get_video_id else 0}\n\n")
            
            f.write("=" * 80 + "\n\n")
            
            # 失败统计
            failed_results = [r for r in results if r['status'] == 'failed']
            if failed_results:
                error_counts = {}
                for result in failed_results:
                    error = result.get('error', '未知错误')
                    error_counts[error] = error_counts.get(error, 0) + 1
                
                f.write("❌ 失败原因统计:\n")
                for error, count in error_counts.items():
                    f.write(f"  {error}: {count} 个\n")
                f.write("\n")
            
            f.write("─" * 80 + "\n\n")
            
            # 详细结果（只记录前100个成功和前50个失败的）
            success_results = [r for r in results if r['status'] == 'success'][:100]
            if success_results:
                f.write("✅ 成功提取的视频 (前100个):\n")
                for result in success_results:
                    f.write(f"  {result['video_key']} -> {os.path.basename(result['saved_image_path'])}")
                    if 'image_size' in result:
                        f.write(f" ({result['image_size'][0]}x{result['image_size'][1]})")
                    f.write("\n")
            
            if failed_results:
                f.write(f"\n❌ 失败的视频 (前50个):\n")
                for result in failed_results[:50]:
                    f.write(f"  {result['video_key']} - {result.get('error', '未知错误')}\n")
        
        print(f"💾 处理结果已保存到: {output_file}")
    
    def debug_mapping(self, sample_keys: list = None):
        """
        调试映射关系
        
        Args:
            sample_keys: 要检查的样本键值列表
        """
        if not self.get_video_id:
            print("❌ 映射尚未初始化")
            return
        
        print("🔍 调试映射关系:")
        print(f"  总映射数量: {len(self.get_video_id)}")
        
        if sample_keys:
            print(f"  检查指定键值:")
            for key in sample_keys:
                if key in self.get_video_id:
                    print(f"    ✅ {key} -> {self.get_video_id[key]}")
                else:
                    print(f"    ❌ {key} -> 不存在")
        else:
            print("  前10个映射:")
            for i, (key, value) in enumerate(list(self.get_video_id.items())[:10]):
                print(f"    {key} -> {value}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简单的LMDB图像提取器 (使用参考代码逻辑)')
    
    parser.add_argument('--txt_path', type=str, 
                       default='/mnt/data/hmdb51/HMDB51_lmdb/annotation_file/hmdb51_train.txt',
                       help='txt文件路径')
    
    parser.add_argument('--lmdb_path', type=str, 
                       default='/mnt/data/hmdb51/HMDB51_lmdb/hmdb51_compressed_frames.lmdb',
                       help='LMDB数据库路径')
    
    parser.add_argument('--output_dir', type=str, 
                       default='/home/stu_b/BIKE/hmdb_qwen_1',
                       help='输出目录路径')
    
    parser.add_argument('--max_videos', type=int, default=None,
                       help='处理的最大视频数量 (默认: 处理所有)')
    
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("🖼️ === 简单LMDB图像提取器 (参考代码逻辑) ===")
    
    # 解析命令行参数
    args = parse_arguments()
    
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
    print(f"  📊 最大处理数: {args.max_videos if args.max_videos else '全部'}")
    print(f"  🐛 调试模式: {'开启' if args.debug else '关闭'}")
    
    try:
        # 初始化提取器
        extractor = SimpleLMDBImageExtractor(output_dir=args.output_dir)
        
        # 如果是调试模式，先初始化LMDB并显示映射信息
        if args.debug:
            print("\n🐛 调试模式：初始化LMDB...")
            extractor.initialize_lmdb(args.lmdb_path)
            
            # 读取txt文件的前几个键值进行调试
            sample_keys = extractor.read_txt_file(args.txt_path)[:5]
            extractor.debug_mapping(sample_keys)
            
            # 关闭LMDB连接
            if extractor.env_i:
                extractor.env_i.close()
            
            print("🐛 调试完成，是否继续处理？(y/n): ", end="")
            if input().lower() != 'y':
                return
        
        # 处理视频
        results = extractor.process_videos(
            txt_path=args.txt_path,
            lmdb_path=args.lmdb_path,
            max_videos=args.max_videos
        )
        
        if results:
            success_count = len([r for r in results if r['status'] == 'success'])
            print(f"📊 最终统计: 成功提取 {success_count} 张图像")
        else:
            print("💥 处理失败")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n💥 程序执行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
