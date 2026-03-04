import torch
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import logging
import os.path
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy
import msgpack
import lmdb
from io import BytesIO
import threading
from collections import OrderedDict
import time
import gc
import psutil
from contextlib import contextmanager
from Coviar.transforms import color_aug

GOP_SIZE = 12

def pil_from_raw_rgb(raw):
    return Image.open(BytesIO(raw)).convert('RGB')

def nparr_from_raw_rgb(raw):
    return np.array(Image.open(BytesIO(raw)).convert('RGB'))

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end

def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos

class PerformanceMonitor:
    """性能监控类"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.lmdb_reads = 0
        self.total_time = 0.0
        self.lmdb_time = 0.0
        self.decode_time = 0.0
        
    def record_cache_hit(self):
        self.cache_hits += 1
        
    def record_cache_miss(self):
        self.cache_misses += 1
        
    def record_lmdb_read(self, duration):
        self.lmdb_reads += 1
        self.lmdb_time += duration
        
    def record_decode_time(self, duration):
        self.decode_time += duration
        
    def get_stats(self):
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hit_rate': hit_rate,
            'total_requests': total_requests,
            'lmdb_reads': self.lmdb_reads,
            'avg_lmdb_time': self.lmdb_time / max(1, self.lmdb_reads),
            'avg_decode_time': self.decode_time / max(1, self.lmdb_reads),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }

class LMDBManager:
    """优化的LMDB管理器 - 单例模式，避免重复连接"""
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, db_path):
        with cls._lock:
            if db_path not in cls._instances:
                cls._instances[db_path] = super().__new__(cls)
                cls._instances[db_path]._initialized = False
            return cls._instances[db_path]
    
    def __init__(self, db_path):
        if self._initialized:
            return
            
        self.db_path = db_path
        self.env = None
        self.txn_cache = {}  # worker_id -> transaction
        self.lock = threading.RLock()
        self._initialized = True
        self._init_lmdb()
    
    def _init_lmdb(self):
        """初始化LMDB环境"""
        try:
            lmdb_config = {
                'readonly': True,
                'lock': False,
                'readahead': True,
                'meminit': False,
                'max_readers': 512,  # 增加读取器数量
                'map_size': 2**42,   # 4TB map size
                'max_dbs': 0,
            }
            
            self.env = lmdb.open(
                self.db_path, 
                subdir=os.path.isdir(self.db_path), 
                **lmdb_config
            )
            
            # 预加载元数据
            with self.env.begin(write=False) as txn:
                self.db_length = msgpack.loads(txn.get(b'__len__'))
                self.db_keys = msgpack.loads(txn.get(b'__keys__'))
                self.db_order = msgpack.loads(txn.get(b'__order__'))
                self.vlen_list = msgpack.loads(txn.get(b'__vlen__'))
                
            self.get_video_id = dict(zip(
                [i for i in self.db_order],
                ['%09d' % i for i in range(len(self.db_order))]
            ))
            
        except Exception as e:
            print(f"Failed to initialize LMDB {self.db_path}: {e}")
            raise
    
    def get_transaction(self):
        """获取当前worker的事务"""
        worker_id = self._get_worker_id()
        
        with self.lock:
            if worker_id not in self.txn_cache:
                try:
                    self.txn_cache[worker_id] = self.env.begin(write=False)
                except Exception as e:
                    print(f"Failed to create transaction for worker {worker_id}: {e}")
                    return None
                    
            return self.txn_cache[worker_id]
    
    def _get_worker_id(self):
        """获取当前worker ID"""
        if hasattr(torch.utils.data, 'get_worker_info'):
            worker_info = torch.utils.data.get_worker_info()
            return worker_info.id if worker_info is not None else 0
        return 0
    
    def read_video_data(self, video_name):
        """读取视频数据"""
        txn = self.get_transaction()
        if txn is None:
            return None
            
        try:
            video_key = self.get_video_id[video_name].encode('ascii')
            raw_data = txn.get(video_key)
            
            if raw_data is None:
                return None
                
            return msgpack.loads(raw_data, raw=True, strict_map_key=False)
        except Exception as e:
            print(f"Error reading video {video_name}: {e}")
            return None
    
    def __del__(self):
        """清理资源"""
        try:
            for txn in self.txn_cache.values():
                txn.abort()
            if self.env:
                self.env.close()
        except:
            pass

class SmartCache:
    """智能缓存系统"""
    def __init__(self, max_size=300, memory_limit_mb=2048):
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.size_tracker = {}  # 跟踪每个缓存项的大小
        self.total_size_mb = 0
        
    def _estimate_size(self, data):
        """估算数据大小（MB）"""
        if isinstance(data, list):
            # 估算视频帧列表的大小
            if len(data) > 0:
                # 假设每帧平均大小
                sample_size = len(data[0]) if hasattr(data[0], '__len__') else 1024*1024
                return len(data) * sample_size / (1024 * 1024)
        return 1  # 默认1MB
    
    def get(self, key):
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # LRU: 移到末尾
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key, value):
        """存储缓存项"""
        value_size = self._estimate_size(value)
        
        with self.lock:
            # 检查是否需要清理空间
            while (len(self.cache) >= self.max_size or 
                   self.total_size_mb + value_size > self.memory_limit_mb):
                if not self.cache:
                    break
                # 移除最老的项
                old_key, old_value = self.cache.popitem(last=False)
                old_size = self.size_tracker.pop(old_key, 1)
                self.total_size_mb -= old_size
            
            self.cache[key] = value
            self.size_tracker[key] = value_size
            self.total_size_mb += value_size
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.size_tracker.clear()
            self.total_size_mb = 0
    
    def get_stats(self):
        """获取缓存统计"""
        with self.lock:
            return {
                'size': len(self.cache),
                'memory_mb': self.total_size_mb,
                'max_size': self.max_size,
                'memory_limit_mb': self.memory_limit_mb
            }


class Video_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, modality='iframe', new_length=1,
                 transform=None, random_shift=True, test_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3,
                 num_sample=1, accumulate=True,
                 iframe_db_path='', mv_db_path='', res_db_path='',
                 gop_size=GOP_SIZE):

        # 原有参数设置
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.modality = modality
        self.seg_length = new_length
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.sample_range = 128
        self.dense_sample = dense_sample
        self.test_clips = test_clips
        self.num_sample = num_sample
        self.accumulate = accumulate
        self._input_size = 224
        self.gop_size = gop_size

        # LMDB路径
        self.iframe_db_path = iframe_db_path
        self.mv_db_path = mv_db_path
        self.residual_db_path = res_db_path

        # 归一化参数
        self.input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self.input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')

        if self.index_bias is None:
            self.index_bias = 1

        # **核心优化：初始化管理器和缓存**
        self.lmdb_managers = {}
        self.cache = SmartCache(max_size=400, memory_limit_mb=3072)  # 3GB缓存
        self.performance_monitor = PerformanceMonitor()

        # 统计信息
        self.load_count = 0
        self.last_stats_print = time.time()
        self._spatial_shape = None

        # self._init_lmdb_managers()
        self.lmdb_managers = {}
        self._parse_list()

    def _get_lmdb_manager(self, modality):
        """
        懒加载获取 LMDB 句柄。
        这保证了只在 DataLoader 的 worker 进程真正需要读数据时，才打开 LMDB 环境。
        """
        if modality not in self.lmdb_managers:
            # 获取和训练代码中同名的 logger (自动输出到同一个日志文件)
            logger = logging.getLogger('BIKE')
            
            db_path = getattr(self, f"{modality}_db_path", "")
            if db_path:
                # 顺便打印出是哪个子进程 (PID) 打开的数据库，方便后续 debug
                logger.info(f"[Worker PID: {os.getpid()}] Initializing LMDB for {modality} at {db_path}")
                self.lmdb_managers[modality] = LMDBManager(db_path)
            else:
                logger.error(f"[Worker PID: {os.getpid()}] ERROR: {modality}_db_path is empty or None!")
                self.lmdb_managers[modality] = None
                
        return self.lmdb_managers[modality]

    def _init_lmdb_managers(self):
        """初始化LMDB管理器"""
        try:
            print("self.iframe_db_path:", self.iframe_db_path)
            print("self.mv_db_path:", self.mv_db_path)
            print("self.residual_db_path:", self.residual_db_path)
            if self.iframe_db_path:
                self.lmdb_managers['iframe'] = LMDBManager(self.iframe_db_path)
            if self.mv_db_path:
                self.lmdb_managers['mv'] = LMDBManager(self.mv_db_path)
            if self.residual_db_path:
                self.lmdb_managers['residual'] = LMDBManager(self.residual_db_path)

            print(f"Initialized LMDB managers for modalities: {list(self.lmdb_managers.keys())}")
        except Exception as e:
            print(f"Failed to initialize LMDB managers: {e}")
            raise

    def _load_video_data_optimized(self, video_name, modality):
        """优化的视频数据加载"""
        cache_key = f"{video_name}_{modality}"

        # 尝试从缓存获取
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            self.performance_monitor.record_cache_hit()
            return cached_data

        self.performance_monitor.record_cache_miss()

        # 从LMDB加载
        start_time = time.time()
        # ✅ 使用懒加载获取 manager
        manager = self._get_lmdb_manager(modality)
        
        # 如果因为路径为空导致 manager 没被成功创建
        start_time = time.time()
        if manager is None:
            logger = logging.getLogger('BIKE')
            logger.error(f"No LMDB manager for modality: {modality}")
            return None

        video_data = manager.read_video_data(video_name)
        lmdb_duration = time.time() - start_time
        self.performance_monitor.record_lmdb_read(lmdb_duration)

        if video_data is not None:
            # 存入缓存
            self.cache.put(cache_key, video_data)

        return video_data

    def _load_frame_from_lmdb(self, video_name, frame_idx, modality):
        """优化的帧加载函数"""
        video_data = self._load_video_data_optimized(video_name, modality)

        if video_data is None:
            return None

        if frame_idx >= len(video_data):
            frame_idx = len(video_data) - 1

        try:
            decode_start = time.time()
            img = nparr_from_raw_rgb(video_data[frame_idx])
            decode_duration = time.time() - decode_start
            self.performance_monitor.record_decode_time(decode_duration)
            return img
        except Exception as e:
            print(f'Error processing frame {frame_idx} from video {video_name}: {e}')
            return None

    def _update_spatial_shape(self, frame):
        if frame is not None:
            self._spatial_shape = frame.shape[:2]

    def _zeros_frame(self, channels):
        if self._spatial_shape is not None:
            h, w = self._spatial_shape
        else:
            h = w = self._input_size
        return np.zeros((h, w, channels), dtype=np.uint8)

    def _prepare_iframe(self, frame):
        if frame is None:
            return self._zeros_frame(3)
        self._update_spatial_shape(frame)
        iframe_bgr = frame[..., ::-1]
        iframe_bgr = color_aug(iframe_bgr)
        return iframe_bgr[..., ::-1]

    def _prepare_mv(self, frame):
        if frame is None:
            return self._zeros_frame(2)
        self._update_spatial_shape(frame)
        mv = frame
        if mv.ndim == 2:
            mv = mv[..., np.newaxis]
        if mv.shape[2] >= 2:
            mv = mv[..., :2]
        else:
            mv = np.repeat(mv[..., :1], 2, axis=2)
        mv = np.clip(mv, 0, 255).astype(np.uint8)
        return mv

    def _prepare_residual(self, frame):
        if frame is None:
            return self._zeros_frame(3)
        self._update_spatial_shape(frame)
        residual = np.clip(frame.astype(np.int32), 0, 255).astype(np.uint8)
        return residual

    def __getitem__(self, index):
        self.load_count += 1

        # 定期打印统计信息
        # if self.load_count % 500 == 0 or time.time() - self.last_stats_print > 60:
        #     self._print_performance_stats()
        #     self.last_stats_print = time.time()

        if not self.test_mode:
            video_record = random.choice(self.video_list)
        else:
            video_record = self.video_list[index]

        video_name = video_record.path
        label = video_record.label
        num_frames = video_record.num_frames

        frames_iframe = []
        frames_mv = []
        frames_residual = []

        for seg in range(self.num_segments):
            # 为每种数据类型分别计算索引
            if not self.test_mode:
                gop_index_iframe, gop_pos_iframe = self._get_train_frame_index(num_frames, seg, "iframe")
                gop_index_res, gop_pos_res = self._get_train_frame_index(num_frames, seg, "residual")
                gop_index_mv, gop_pos_mv = self._get_train_frame_index(num_frames, seg, "mv")
            else:
                gop_index_iframe, gop_pos_iframe = self._get_test_frame_index(num_frames, seg, "iframe")
                gop_index_res, gop_pos_res = self._get_test_frame_index(num_frames, seg, "residual")
                gop_index_mv, gop_pos_mv = self._get_test_frame_index(num_frames, seg, "mv")

            # 使用各自的索引加载数据
            frame_idx_iframe = gop_index_iframe * self.gop_size + gop_pos_iframe
            frame_idx_res = gop_index_res * self.gop_size + gop_pos_res
            frame_idx_mv = gop_index_mv * self.gop_size + gop_pos_mv

            iframe_raw = self._load_frame_from_lmdb(video_name, frame_idx_iframe, 'iframe')
            residual_raw = self._load_frame_from_lmdb(video_name, frame_idx_res, 'residual')
            mv_raw = self._load_frame_from_lmdb(video_name, frame_idx_mv, 'mv')

            # __getitem__ 中的异常处理部分：
            if iframe_raw is None or mv_raw is None or residual_raw is None:
                logger = logging.getLogger('BIKE')
                logger.warning(f'Error: loading video {video_name} failed. Using zero padding.')
                
                # 修复上一次的致命 Bug：一定要赋值给 _raw
                iframe_raw = np.zeros((256, 256, 3), dtype=np.uint8)
                mv_raw = np.zeros((256, 256, 3), dtype=np.uint8)
                residual_raw = np.zeros((256, 256, 3), dtype=np.uint8)

            frames_iframe.append(iframe_raw)
            frames_mv.append(mv_raw)
            frames_residual.append(residual_raw)

        if self.transform is not None:
            frames_iframe = self.transform(frames_iframe)
            frames_mv = self.transform(frames_mv)
            frames_residual = self.transform(frames_residual)

        frames_iframe = np.array(frames_iframe)
        frames_mv = np.array(frames_mv)
        frames_residual = np.array(frames_residual)

        frames_iframe = np.transpose(frames_iframe, (0, 3, 1, 2))
        frames_mv = np.transpose(frames_mv, (0, 3, 1, 2))
        frames_residual = np.transpose(frames_residual, (0, 3, 1, 2))

        input_iframe = torch.from_numpy(frames_iframe).float() / 255.0
        input_mv = torch.from_numpy(frames_mv).float() / 255.0
        input_residual = torch.from_numpy(frames_residual).float() / 255.0

        input_iframe = (input_iframe - self.input_mean) / self.input_std
        input_residual = (input_residual - 0.5) / self.input_std
        input_mv = (input_mv - 0.5)
        return input_iframe, input_mv,input_residual , label



    def _print_performance_stats(self):
        """打印性能统计信息"""
        perf_stats = self.performance_monitor.get_stats()
        cache_stats = self.cache.get_stats()

        print(f"\n=== Dataset Performance Stats (Load Count: {self.load_count}) ===")
        print(f"Cache Hit Rate: {perf_stats['cache_hit_rate']:.2%}")
        print(f"Total Requests: {perf_stats['total_requests']}")
        print(f"LMDB Reads: {perf_stats['lmdb_reads']}")
        print(f"Avg LMDB Time: {perf_stats['avg_lmdb_time']:.4f}s")
        print(f"Avg Decode Time: {perf_stats['avg_decode_time']:.4f}s")
        print(f"Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"Cache Memory: {cache_stats['memory_mb']:.1f}/{cache_stats['memory_limit_mb']:.1f} MB")
        print(f"Process Memory: {perf_stats['memory_usage_mb']:.1f} MB")
        print("=" * 60)

    def get_performance_stats(self):
        """获取性能统计信息"""
        return {
            'performance': self.performance_monitor.get_stats(),
            'cache': self.cache.get_stats(),
            'load_count': self.load_count
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        gc.collect()

    # 保持原有的其他方法不变
    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3:
            if not self.test_mode:
                tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _get_train_frame_index(self, num_frames, seg,modality):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self.num_segments, seg,
                                                 representation=modality)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, modality)

    def _get_test_frame_index(self, num_frames, seg,modality):
        if modality in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self.num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if modality in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, modality)

    def __len__(self):
        return len(self.video_list)

    def __del__(self):
        """清理资源"""
        try:
            self.cache.clear()
            # LMDB管理器会自动清理
        except:
            pass
