import torch
import torch.utils.data as data
import decord
import matplotlib.pyplot as plt
import os
import os.path
import json
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy
from coviar import get_num_frames
from coviar import load
from Coviar.transforms import color_aug


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
    """根据不同的representation类型计算segment范围"""
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frame.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation, GOP_SIZE):
    """根据不同的representation类型计算GOP位置"""
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class Video_compress_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file, 
                 num_segments=1, modality='compress', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3,
                 num_sample=1, accumulate=False, GOP_SIZE=12,
                 text_map_path=None):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
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
        self.GOP_SIZE = GOP_SIZE
        self.return_description = text_map_path is not None
        
        self.input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self.input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self._load_text_map(text_map_path)

    def _load_text_map(self, text_map_path):
        self.text_map = {}
        if not text_map_path:
            return
        try:
            with open(text_map_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for key, value in raw.items():
                desc = self._extract_description(value)
                if desc:
                    self.text_map[key] = desc
        except Exception as e:
            print(f"Failed to load text descriptions from {text_map_path}: {e}")
            self.text_map = {}

    @staticmethod
    def _extract_description(value):
        if isinstance(value, dict):
            for field in ['semantic_description', 'description', 'caption', 'text']:
                desc = value.get(field)
                if desc:
                    if isinstance(desc, list):
                        return " ".join(str(x) for x in desc)
                    return str(desc)
            return ""
        if isinstance(value, list):
            return " ".join(str(x) for x in value)
        if isinstance(value, str):
            return value
        return str(value)

    def _get_description(self, video_record):
        key = os.path.splitext(video_record.path)[0]
        description = self.text_map.get(key, "") if hasattr(self, 'text_map') else ""
        if description:
            return description
        # 默认回退为简单描述，避免空字符串
        action_name = key.split('/')[-1].replace('_', ' ')
        return f"This video shows {action_name}."

    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        if self.modality in ['RGB', 'video']:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if len(tmp[0]) == 3:
                if not self.test_mode:
                    tmp = [item for item in tmp if int(item[1]) >= 8]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))
        elif self.modality in ['iframe', 'mv', 'residual', 'compress']:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if len(tmp[0]) == 3:
                if not self.test_mode:
                    tmp = [item for item in tmp if int(item[1]) >= 8]
            tmp = [(item[0] + '.mp4', item[1], item[2] if len(item) > 2 else None) for item in tmp]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % len(video_list)
            return np.array(offsets) + self.index_bias
        else:
            seg_size = float(len(video_list) - 1) / self.num_segments
            offsets = []
            for i in range(self.num_segments):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                offsets.append(random.randint(start, end))
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            t_stride = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        else:
            tick = len(video_list) / float(self.num_segments)
            offsets = [int(tick * x) % len(video_list) for x in range(self.num_segments)]
            return np.array(offsets) + self.index_bias

    def _get_test_indices(self, video_list):
        if self.dense_sample:
            num_clips = self.test_clips
            sample_pos = max(0, len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_list = [clip_idx * math.floor(sample_pos / (num_clips -1)) for clip_idx in range(num_clips)]
            base_offsets = np.arange(self.num_segments) * interval
            offsets = []
            for start_idx in start_list:
                offsets.extend((base_offsets + start_idx) % len(video_list))
            return np.array(offsets) + self.index_bias
        else:
            num_clips = self.test_clips
            tick = len(video_list) / float(self.num_segments)
            start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [
                    int(start_idx + tick * x) % len(video_list)
                    for x in range(self.num_segments)
                ]
            return np.array(offsets) + self.index_bias

    def _decord_decode(self, video_path):
        try:
            container = decord.VideoReader(video_path)
        except Exception as e:
            print("Failed to decode {} with exception: {}".format(video_path, e))
            return None
        return container

    def _get_train_frame_index(self, num_frames, seg, representation, GOP_SIZE):
        """为特定的representation类型计算训练时的帧索引"""
        seg_begin, seg_end = get_seg_range(num_frames, self.num_segments, seg, representation)
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, representation, GOP_SIZE)

    def _get_test_frame_index(self, num_frames, seg, representation, GOP_SIZE):
        """为特定的representation类型计算测试时的帧索引"""
        temp_num_frames = num_frames
        if representation in ['mv', 'residual']:
            temp_num_frames -= 1

        seg_size = float(temp_num_frames - 1) / self.num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, representation, GOP_SIZE)

    def __getitem__(self, index):
        if self.modality in ['RGB', 'video']:
            if self.modality == 'video':
                _num_retries = 10
                for i_try in range(_num_retries):
                    record = copy.deepcopy(self.video_list[index])
                    directory = os.path.join(self.root_path, record.path)
                    video_list = self._decord_decode(directory)
                    if video_list is None:
                        print("Failed to decode video idx {} from {}; trial {}".format(
                            index, directory, i_try))
                        index = random.randint(0, len(self.video_list))
                        continue
                    break
            else:
                record = self.video_list[index]
                video_list = os.listdir(os.path.join(self.root_path, record.path))

            if not self.test_mode:
                segment_indices = self._sample_indices(video_list) if self.random_shift else self._get_val_indices(video_list) 
            else:
                segment_indices = self._get_test_indices(video_list)

            return self.get(record, video_list, segment_indices)
        
        elif self.modality in ['iframe', 'mv', 'residual', 'compress']:
            if not self.test_mode:
                video_record = random.choice(self.video_list)
                video_path = os.path.join(self.root_path, video_record.path)
                label = video_record.label
                num_frames = video_record.num_frames
            else:
                video_record = self.video_list[index]
                video_path = os.path.join(self.root_path, video_record.path)
                label = video_record.label
                num_frames = video_record.num_frames

            frames_iframe = []
            frames_mv = []
            frames_residual = []

            for seg in range(self.num_segments):
                # 为每种数据类型分别计算帧索引
                if not self.test_mode:
                    # 训练模式：为每种类型分别计算索引
                    gop_index_iframe, gop_pos_iframe = self._get_train_frame_index(num_frames, seg, 'iframe', self.GOP_SIZE)
                    gop_index_mv, gop_pos_mv = self._get_train_frame_index(num_frames, seg, 'mv', self.GOP_SIZE)
                    gop_index_residual, gop_pos_residual = self._get_train_frame_index(num_frames, seg, 'residual', self.GOP_SIZE)
                else:
                    # 测试模式：为每种类型分别计算索引
                    gop_index_iframe, gop_pos_iframe = self._get_test_frame_index(num_frames, seg, 'iframe', self.GOP_SIZE)
                    gop_index_mv, gop_pos_mv = self._get_test_frame_index(num_frames, seg, 'mv', self.GOP_SIZE)
                    gop_index_residual, gop_pos_residual = self._get_test_frame_index(num_frames, seg, 'residual', self.GOP_SIZE)

                # 加载iframe
                iframe = load(video_path, gop_index_iframe, gop_pos_iframe, 0, self.accumulate)
                if iframe is None:
                    print('Error: loading iframe from video %s failed.' % video_path)
                    iframe = np.zeros((256, 256, 3))
                else:
                    iframe = color_aug(iframe)
                    iframe = iframe[..., ::-1]
                frames_iframe.append(iframe)

                # 加载mv
                mv = load(video_path, gop_index_mv, gop_pos_mv, 1, self.accumulate)
                if mv is None:
                    print('Error: loading mv from video %s failed.' % video_path)
                    mv = np.zeros((256, 256, 2))
                else:
                    mv += 128
                    mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
                frames_mv.append(mv)

                # 加载residual
                residual = load(video_path, gop_index_residual, gop_pos_residual, 2, self.accumulate)
                if residual is None:
                    print('Error: loading residual from video %s failed.' % video_path)
                    residual = np.zeros((256, 256, 3))
                else:
                    residual += 128
                    residual = (np.minimum(np.maximum(residual, 0), 255)).astype(np.uint8)
                frames_residual.append(residual)

            # 应用变换
            frames_iframe = self.transform(frames_iframe)
            frames_mv = self.transform(frames_mv)
            frames_residual = self.transform(frames_residual)

            # 转换为numpy数组并调整维度
            frames_iframe = np.array(frames_iframe)
            frames_mv = np.array(frames_mv)
            frames_residual = np.array(frames_residual)

            frames_iframe = np.transpose(frames_iframe, (0, 3, 1, 2))
            frames_mv = np.transpose(frames_mv, (0, 3, 1, 2))
            frames_residual = np.transpose(frames_residual, (0, 3, 1, 2))

            # 转换为tensor并归一化
            input_iframe = torch.from_numpy(frames_iframe).float() / 255.0
            input_mv = torch.from_numpy(frames_mv).float() / 255.0
            input_residual = torch.from_numpy(frames_residual).float() / 255.0

            # 应用不同的归一化策略
            input_iframe = (input_iframe - self.input_mean) / self.input_std
            input_residual = (input_residual - 0.5) / self.input_std
            input_mv = (input_mv - 0.5)

            if self.return_description:
                description = self._get_description(video_record)
                return input_iframe, input_mv, input_residual, label, description
            else:
                return input_iframe, input_mv, input_residual, label

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def get(self, record, video_list, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            if self.modality == 'video':
                seg_imgs = [Image.fromarray(video_list[p - 1].asnumpy()).convert('RGB')]
            else:
                seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
            if p < len(video_list):
                p += 1
        if self.num_sample > 1:
            frame_list = []
            label_list = []
            for _ in range(self.num_sample):
                process_data, record_label = self.transform((images, record.label))
                frame_list.append(process_data)
                label_list.append(record_label)
            return frame_list, label_list
        else:
            process_data, record_label = self.transform((images, record.label))
            return process_data, record_label

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    transform_train = None
    train_data = Video_compress_dataset(
        '/datasets/hmdb51/mpeg4_videos', 
        '/home/mmstu_b/gmk/BIKE/lists/hmdb51/train_rgb_split_1.txt',
        '/home/mmstu_b/gmk/BIKE/lists/hmdb51/hmdb51_labels.csv', 
        num_segments=16,
        modality='compress',  # 改为compress
        transform=transform_train, 
        random_shift=True
    )
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

    for i, (iframe, mv, residual, label) in enumerate(data_loader):
        print(f"iframe shape: {iframe.shape}")
        print(f"mv shape: {mv.shape}")
        print(f"residual shape: {residual.shape}")
        print(f"label shape: {label.shape}")
        break
