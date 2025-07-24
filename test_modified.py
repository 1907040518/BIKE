
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
import clip

import yaml
from dotmap import DotMap
from datasets.video import Video_dataset
from datasets.compress_3 import Video_compress_dataset  # 🆕 添加压缩数据集
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from Coviar.transforms import get_compress_augmentation  # 🆕 添加压缩数据增强
from modules.video_clip import video_header
from modules.text_prompt import text_prompt
from utils.utils import gather_labels  # 🆕 添加标签收集函数


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    parser.add_argument('--no-accumulation', action='store_true',  # 🆕 添加accumulation参数
                    help='disable accumulation of motion vectors and residuals.')
    args = parser.parse_args()
    return args

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

# 🆕 添加AllGather类（从训练代码复制）
class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

allgather = AllGather.apply

def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # 🔧 修改模型加载参数以匹配训练代码
    model, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        Block=config.network.Block,  # 🆕 添加Block参数
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st=config.network.joint_st)  # 🆕 添加joint_st参数

    # 🔧 创建双头结构
    video_head = video_header(
        config.network.sim_header,
        config.network.S_Align,  # 🆕 使用S_Align
        clip_state_dict)

    mv_head = video_header(  # 🆕 添加MV头
        config.network.sim_header,
        config.network.M_Align,  # 🆕 使用M_Align
        clip_state_dict)

    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()

    # 🔧 修改数据增强以支持压缩数据
    if config.data.modality in ['mv', 'residual', 'iframe']:
        transform_val = get_compress_augmentation(False, config)
    else:
        # 原始的数据增强逻辑
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]

        # rescale size
        if 'something' in config.data.dataset:
            scale_size = (240, 320)
        else:
            scale_size = 256 if config.data.input_size == 224 else config.data.input_size

        # crop size
        input_size = config.data.input_size

        # control the spatial crop
        if args.test_crops == 1: # one crop
            cropping = torchvision.transforms.Compose([
                GroupScale(scale_size),
                GroupCenterCrop(input_size),
            ])
        elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(
                    crop_size=input_size,
                    scale_size=scale_size,
                    flip=False)
            ])
        elif args.test_crops == 5:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupOverSample(
                    crop_size=input_size,
                    scale_size=scale_size,
                    flip=False)
            ])
        elif args.test_crops == 10:
            cropping = torchvision.transforms.Compose([
                GroupOverSample(
                    crop_size=input_size,
                    scale_size=scale_size,
                )
            ])
        else:
            raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))

        transform_val = torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ])

    # 🔧 根据模态选择数据集
    if config.data.modality in ['RGB', 'video']:
        if config.data.dataset == 'charades':
            from datasets.charades import Video_dataset
            val_data = Video_dataset(
                config.data.val_root, config.data.val_list, config.data.label_list,
                random_shift=False, num_segments=config.data.num_segments,
                modality=config.data.modality,
                image_tmpl=config.data.image_tmpl,
                transform=transform_val, test_mode=True, dense_sample=config.data.dense)
        else:
            val_data = Video_dataset(
                config.data.val_root, config.data.val_list, config.data.label_list,
                random_shift=False, num_segments=config.data.num_segments,
                modality=config.data.modality,
                image_tmpl=config.data.image_tmpl,
                transform=transform_val, test_mode=True, dense_sample=args.dense,
                test_clips=args.test_clips)
    elif config.data.modality in ['iframe', 'mv', 'residual']:  # 🆕 支持压缩数据
        val_data = Video_compress_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=args.dense, 
            accumulate=(not args.no_accumulation), test_mode=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)

    # 🔧 修改权重加载以支持双头结构
    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        if dist.get_rank() == 0:
            print('load model: epoch {}'.format(checkpoint['epoch']))

        model.load_state_dict(update_dict(checkpoint['model_state_dict']))
        video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
        
        # 🆕 加载MV头权重（如果存在）
        if 'mv_head_state_dict' in checkpoint:
            mv_head.load_state_dict(update_dict(checkpoint['mv_head_state_dict']))
        else:
            # 如果没有单独的MV头权重，使用相同的fusion权重
            mv_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            
        del checkpoint

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        if config.network.sim_header != "None":
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])
            mv_head = DistributedDataParallel(mv_head.cuda(), device_ids=[args.gpu])  # 🆕 添加MV头分布式

    # 🔧 修改文本提示生成
    classes, n_class = text_prompt(val_data, config)  # 🆕 匹配训练代码的返回格式

    # 🔧 根据数据集选择验证函数
    if config.data.dataset == 'charades':
        prec1 = validate_mAP(
            val_loader, classes, device,
            model, video_head, mv_head, config, n_class, args.test_crops, args.test_clips)
    else:
        prec1 = validate(
            val_loader, classes, device,
            model, video_head, mv_head, config, n_class, args.test_crops, args.test_clips)
    return


def validate(val_loader, classes, device, model, video_head, mv_head, config, n_class, test_crops, test_clips):
    """🔧 完全重写验证函数以匹配训练代码的三模态处理"""
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    video_head.eval()
    mv_head.eval()  # 🆕 设置MV头为评估模式
    
    proc_start_time = time.time()
    sim_logits = []   
    labels = []   
    
    with torch.no_grad():
        text_inputs = classes.to(device)  # [n_cls, 77]
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)
        
        for i, (image, mv, residual, class_id) in enumerate(val_loader):  # 🆕 三模态输入
            batch_size = class_id.numel()
            num_crop = test_crops
            num_crop *= test_clips

            class_id = class_id.to(device)
            n_seg = config.data.num_segments
            
            # 🔧 按照训练代码处理三模态数据
            image = image.view((-1, n_seg, 3) + image.size()[-2:])  # b t 3 h w
            mv = mv.view((-1, n_seg, 2) + mv.size()[-2:])          # b t 2 h w
            residual = residual.view((-1, n_seg, 3) + residual.size()[-2:])  # b t 3 h w
            
            b, t, c_i, h, w = image.size()
            b, t, c_m, h, w = mv.size()
            
            # 展平处理
            image_input = image.to(device).view(-1, c_i, h, w)
            mv_input = mv.to(device).view(-1, c_m, h, w) 
            residual_input = residual.to(device).view(-1, c_i, h, w)
            
            # 🔧 使用训练代码的特征提取方式
            image_features, mv_features, res_features = model.module.encode_image(
                image_input, mv_input, residual_input)
            
            # 🔧 按照训练代码进行特征融合
            weights = F.softmax(model.module.beta, dim=0)
            merged_feats = weights[0] * image_features + weights[1] * res_features
            
            # 重塑维度
            mv_features = mv_features.view(b, t, -1)
            merged_feats = merged_feats.view(b, t, -1)
            
            cnt_time = time.time() - proc_start_time
            
            # 🔧 使用双头计算相似度
            similarity = video_head(merged_feats, text_features, cls_feature)
            similarity_mv = mv_head(mv_features, text_features, cls_feature)
            
            # 🔧 按照训练代码进行加权融合
            combined_similarity = 0.4 * similarity + 0.6 * similarity_mv
            final_similarity = combined_similarity
            
            # 处理多crop和多clip
            final_similarity = F.softmax(final_similarity, -1)
            final_similarity = final_similarity.reshape(batch_size, num_crop, -1).mean(1)
            final_similarity = final_similarity.view(batch_size, -1, n_class).softmax(dim=-1)
            final_similarity = final_similarity.mean(dim=1, keepdim=False)

            # 🔧 处理特定数据集的保存需求
            if 'anet' in config.data.dataset:
                sim_logits.append(concat_all_gather(final_similarity))
                labels.append(concat_all_gather(class_id))

            prec = accuracy(final_similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i+1) / (batch_size * dist.get_world_size())
                print(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), runtime=runtime, top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        print('-----Evaluation is finished------')
        print('Overall Prec@1 {:.03f}% Prec@5 {:.03f}%'.format(top1.avg, top5.avg))

    # 🔧 处理mAP计算
    if 'anet' in config.data.dataset:
        sim, gt = sim_logits[0], labels[0]
        for i in range(1, len(sim_logits)): 
            sim = torch.cat((sim, sim_logits[i]), 0)
            gt = torch.cat((gt, labels[i]), 0)

        if dist.get_rank() == 0:
            from utils.utils import mean_average_precision
            mAP = mean_average_precision(sim, gt)
            print('Overall mAP: {:.03f}%'.format(mAP[1].item()))

    return top1.avg


def validate_mAP(val_loader, classes, device, model, video_head, mv_head, config, n_class, test_crops, test_clips):
    """🆕 为Charades数据集添加mAP验证函数"""
    from torchnet import meter
    maper = meter.mAPMeter()
    model.eval()
    video_head.eval()
    mv_head.eval()

    with torch.no_grad():
        text_inputs = classes.to(device)
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)
        
        for i, (image, mv, residual, class_id) in enumerate(val_loader):
            # 按照训练代码处理数据
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
            residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])

            b, t, c_i, h, w = image.size()
            class_id = class_id.to(device)

            image_input = image.to(device).view(-1, c_i, h, w)
            mv_input = mv.to(device).view(-1, 2, h, w)
            residual_input = residual.to(device).view(-1, 3, h, w)

            # 特征提取和融合
            image_features, mv_features, res_features = model.module.encode_image(
                image_input, mv_input, residual_input)
            
            weights = F.softmax(model.module.beta, dim=0)
            merged_features = weights[0] * image_features + weights[1] * res_features
            
            merged_features = merged_features.view(b, t, -1)
            mv_features = mv_features.view(b, t, -1)

            # 双头计算相似度
            video_similarity = video_head(merged_features, text_features, cls_feature)
            mv_similarity = mv_head(mv_features, text_features, cls_feature)
            combined_similarity = 0.4 * video_similarity + 0.6 * mv_similarity

            combined_similarity = combined_similarity.view(b, -1, n_class).softmax(dim=-1)
            combined_similarity = combined_similarity.mean(dim=1, keepdim=False)
            combined_similarity = F.softmax(combined_similarity, dim=1)

            output = allgather(combined_similarity)
            labels_gathered = gather_labels(class_id)
            maper.add(output, labels_gathered)

            if i % config.logging.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}], mAP: {maper.value().numpy():.3f}%')

    final_mAP = maper.value().numpy() * 100
    print(f'Testing Results mAP === {final_mAP:.3f}%')
    return final_mAP


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output.cpu()


if __name__ == '__main__':
    args = get_parser()
    main(args)
