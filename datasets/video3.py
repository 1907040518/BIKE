import torch
import torchvision
import torch.utils.data as data
import decord
import matplotlib.pyplot as plt
import os
import os.path
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

GOP_SIZE = 12

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
# class VideoRecord(object):
#     def __init__(self, row, data_root):
#         self._data_root = data_root
#         self._data = row

#     @property
#     def path(self):
#         # 获取完整路径
#         return os.path.join(self._data_root, self._data[0])

#     @property
#     def num_frames(self):
#         # 动态获取帧数
#         return get_num_frames(self.path)

#     @property
#     def label(self):
#         return int(self._data[-1])

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
        # Exclude the 0-th frame, because it's an I-frmae.
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

import numpy as np
import cv2
from torchvision.utils import flow_to_image
import torch

def flow_to_image_torch(flow):
    flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
    flow_im = flow_to_image(flow)
    img = np.transpose(flow_im.numpy(), [1, 2, 0])
    print(img.shape)
    return img

import numpy as np
import cv2

def visualize_flow(flow):
    """
    将光流场可视化为颜色场。
    :param flow: 光流场，形状为 (H, W, 2)，其中每个像素包含 (dx, dy)。
    :return: 可视化的光流图，形状为 (H, W, 3)。
    """
    # 确保输入的光流场是 float32 类型
    flow = flow.astype(np.float32)

    # 将光流场转换为极坐标
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 创建 HSV 图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # 饱和度设置为最大值
    
    # 色调（H）表示运动方向，映射到 0-180 度
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # 亮度（V）表示运动大小，归一化到 0-255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # 将 HSV 图像转换为 BGR 图像
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr




class Video_dataset(data.Dataset):
    # modality='RGB' 'iframe' 'mv' 'res' 'videos' 'compress'
    def __init__(self, root_path, list_file, labels_file, 
                 num_segments=1, modality='RGB', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3,
                 num_sample=1, accumulate = True):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.sample_range = 128
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.test_clips = test_clips
        self.num_sample = num_sample
        self.accumulate = accumulate
        self._input_size =224
        
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
        # self._load_list(list_file)


    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
        

    def _load_list(self, video_list):
        #创建一个空列表 self._video_list，用于存储视频文件的信息
        self.video_list = []
        #打开 video_list 文本文件，并使用 with open(video_list, 'r') as f: 的上下文管理器来确保文件在使用完毕后自动关闭。
        with open(video_list, 'r') as f:
            for line in f:
                #对于每一行，使用 .strip() 方法去除首尾的空白字符，并使用 .split() 方法将行拆分成多个部分，按空格进行分割。
                #拆分成三个部分，视频，_，标签
                video, _, label = line.strip().split()
                #根据视频文件名构造视频文件的完整路径，通过 os.path.join(self._data_root, video[:-4] + '.mp4') 拼接文件路径。
                #[:-4] 是对字符串进行切片操作，截取除去最后四个字符（也就是文件扩展名）之外的部分。
                #这种方式可以确保在不同操作系统上使用正确的路径分隔符，并且能够根据输入的路径和文件名创建正确的文件路径，方便处理视频文件。
                # print('----------------------------------------------------------------')
                # print(self.root_path)
                # print(video)
                video_path = os.path.join(self.root_path, video + '.mp4' )
                # print(video_path)
                # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
               
                #创建一个元组 (video_path, int(label), get_num_frames(video_path))，添加到列表中,表示一个视频信息
                #video_path 是视频文件的完整路径。  int(label) 是标签，转换为整数类型 
                #get_num_frames(video_path) 是调用了一个名为 get_num_frames() 的函数(coviar导入），用于获取视频文件的帧数。
                self.video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))
        #输出加载了多少视频信息
        print('%d videos loaded.' % len(self.video_list))
    def _parse_list(self):
        if self.modality in ['RGB', 'video']:
            # check the frame number is large >3:  查看训练集和验证集视频数量并输出
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
                if not self.test_mode:
                    tmp = [item for item in tmp if int(item[1]) >= 8]
            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))
        elif self.modality in ['iframe', 'mv', 'residual']:
            # check the frame number is large >3:  查看训练集和验证集视频数量并输出
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if len(tmp[0]) == 3:  # skip remove_missin for decording "raw_video label" type dataset_config
                if not self.test_mode:
                    tmp = [item for item in tmp if int(item[1]) >= 8]
            tmp = [(item[0] + '.mp4', item[1], item[2] if len(item) > 2 else None) for item in tmp]

            self.video_list = [VideoRecord(item) for item in tmp]
            print('video number:%d' % (len(self.video_list)))
    # def _parse_list(self):
    #     if self.modality in ['RGB', 'video']:
    #         # check the frame number is large >3:  查看训练集和验证集视频数量并输出
    #         tmp = [x.strip().split(' ') for x in open(self.list_file)]
    #         if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
    #             if not self.test_mode:
    #                 tmp = [item for item in tmp if int(item[1]) >= 8]
    #         self.video_list = [VideoRecord(item) for item in tmp]
    #         print('video number:%d' % (len(self.video_list)))
    #     elif self.modality in ['iframe', 'mv', 'residual']:
    #         # check the frame number is large >3:  查看训练集和验证集视频数量并输出
    #         tmp = [x.strip().split(' ') for x in open(self.list_file)]

    #         video_records = []
    #         for item in tmp:
    #             video_record = VideoRecord(item, self.root_path)
    #             if not self.test_mode:
    #                 # 仅在训练模式下，动态过滤帧数小于 8 的视频
    #                 if video_record.num_frames >= 8:
    #                     video_records.append(video_record)
    #             else:
    #                 # 测试模式下不过滤
    #                 video_records.append(video_record)          
    #         self.video_list = video_records
    #         print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % len(video_list)
            return np.array(offsets) + self.index_bias
        else:
            # 取num_segments帧
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
            # multi-clip for dense sampling
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
            # multi-clip for uniform sampling
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
            print("Failed to decode {} with exception: {}".format(
                video_path, e))
            return None
        
        return container


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

    def __getitem__(self, index):
        # decode frames to video_list
        if not self.test_mode:
            video_record = random.choice(self.video_list)  # 从 self._video_list 中随机选择一个视频路径、标签和帧数。
            video_path = os.path.join(self.root_path, video_record.path)
            label = video_record.label
            num_frames = video_record.num_frames
        else:
            video_path = os.path.join(self.root_path, self.video_list[index].path)
            label = self.video_list[index].label
            num_frames = self.video_list[index].num_frames

        iframe = []
        res = []
        mv = []
        for seg in range(self.num_segments):   # 获取num segment

            if not self.test_mode:
                gop_index_iframe, gop_pos_iframe = self._get_train_frame_index(num_frames, seg,"iframe")
                gop_index_res, gop_pos_res = self._get_train_frame_index(num_frames, seg,"residual")
                gop_index_mv, gop_pos_mv = self._get_train_frame_index(num_frames, seg,"mv")
            else:
                gop_index_iframe, gop_pos_iframe = self._get_test_frame_index(num_frames, seg,"iframe")
                gop_index_res, gop_pos_res = self._get_test_frame_index(num_frames, seg,"residual")
                gop_index_mv, gop_pos_mv = self._get_test_frame_index(num_frames, seg,"mv")

            img_iframe = load(video_path, gop_index_iframe, gop_pos_iframe,
                    0, self.accumulate)
            img_res = load(video_path, gop_index_res, gop_pos_res,
                    2, self.accumulate)
            img_mv = load(video_path, gop_index_mv, gop_pos_mv,
                    1, self.accumulate)
            
            if img_iframe is None:
                print('Error: loading video %s failed.' % video_path)
                img_mv = np.zeros((256, 256, 2))
                img_res = np.zeros((256, 256, 3))
                img_iframe = np.zeros((256, 256, 3))
            else:
                
                img_mv = clip_and_scale(img_mv, 20)
                img_mv += 128
                img_mv = (np.minimum(np.maximum(img_mv, 0), 255)).astype(np.uint8)
            
                img_res += 128
                img_res = (np.minimum(np.maximum(img_res, 0), 255)).astype(np.uint8)
                
            
            img_iframe = color_aug(img_iframe)

            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img_iframe = img_iframe[..., ::-1]
            
            iframe.append(img_iframe)
            res.append(img_res)
            mv.append(img_mv)
##############################################################################
        if isinstance(self.transform, dict):
            # 是字典类型
            iframe = self.transform['iframe'](iframe)
            mv = self.transform['mv'](mv)
            res = self.transform['residual'](res)
        else:
            # 不是字典类型（假设是Compose对象）
            iframe = self.transform(iframe)
            mv = self.transform(mv)
            res = self.transform(res)
        iframe = np.array(iframe)
        res = np.array(res)
        mv = np.array(mv)
##############################################################################
   

        iframe = np.transpose(iframe, (0, 3, 1, 2))
        iframe = torch.from_numpy(iframe).float() / 255.0
        res = np.transpose(res, (0, 3, 1, 2))
        res = torch.from_numpy(res).float() / 255.0
        mv = np.transpose(mv, (0, 3, 1, 2))
        mv = torch.from_numpy(mv).float() / 255.0

        iframe = (iframe - self.input_mean) / self.input_std

        res = (res - 0.5) / self.input_std

        mv = (mv - 0.5)
        # print(iframe.shape)
        # print(res.shape)
        # print(mv.shape)
        return iframe,res,mv, label


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
    import yaml
    from dotmap import DotMap
    from Coviar.transforms import get_compress_augmentation
    
    path = "/home/stu_a/BIKE-main/configs/hmdb51/hmdb_k400_finetune.yaml"
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print(config)
    config = DotMap(config)
    transform_train = get_compress_augmentation(True, config)

    train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality="mv",
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense)
    data_loader = data.DataLoader(
        train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    for i, inputs in enumerate(data_loader):
        flow_im = flow_to_image(inputs[0][0][0])  # inputs[0] is the video tensor
        img = np.transpose(flow_im.numpy(), [1, 2, 0])
        cv2.imwrite(f"/home/stu_a/BIKE-main/datasets/img/mv_{i}.png", img)
        # 打印图像的形状
        print(img.shape)
        exit()
        pass


