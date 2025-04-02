import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import numpy as np

from utils.utils import init_distributed_mode, epoch_saving, best_saving, AverageMeter, reduce_tensor, accuracy, create_logits, gen_label, gather_labels
from utils.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress

from modules.video_clip import video_header
from utils.NCELoss import NCELoss, DualLoss
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler
from modules.text_prompt import text_prompt

from Coviar.transforms import get_compress_augmentation, GroupCenterCrop, GroupScale
from X_CLIP.models.prompt import VideoSpecificPrompt
from X_CLIP.models.prompt import Video_Prompt
torch.autograd.set_detect_anomaly(True)  # 在代码开头启用
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

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="fp32",
        help="Floating point precition."
    )        
    parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')                
    args = parser.parse_args()
    return args



def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)   # 初始化分布式训练环境函数
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = os.path.join(config['data']['output_path'], config['data']['dataset'], config['network']['arch'] , args.log_time)


    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        # 获取当前执行脚本的文件名
        current_script_name = os.path.basename(__file__)
        shutil.copy(current_script_name, working_dir)


    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'BIKE')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))



    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # get fp16 model and weight
    # model: 这将是一个可用于前向推理或继续训练的 CLIP 模型实例。你可以使用这个模型输入图像和文本进行特征提取、相似度计算等任务。
    # clip_state_dict: 包含了模型当前的权重和偏置，你可以使用这个字典在训练过程中更新模型的参数，或者在保存和加载模型时使用。
    model, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu',jit=False,
        internal_modeling=config.network.tm,
        Block=config.network.Block,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st = config.network.joint_st) # Must set jit=False for training  ViT-B/32
    
    if config.data.modality in ['mv', 'residual', 'iframe']:
        transform_train = get_compress_augmentation(True, config)
        transform_val = get_compress_augmentation(False, config)
        
    else:
        transform_train = get_augmentation(True, config)
        transform_val = get_augmentation(False, config)


    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))


    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict)


    mv_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict)
    
    video_prompt = Video_Prompt(clip_state_dict)


    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()

    
    if config.data.modality in ['RGB', 'video']:
        if config.data.dataset == 'charades':
            from datasets.charades import Video_dataset
            train_data = Video_dataset(
                config.data.train_root, config.data.train_list,
                config.data.label_list, num_segments=config.data.num_segments,
                modality=config.data.modality,
                image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
                transform=transform_train, dense_sample=config.data.dense,
                fps=config.data.fps)
            val_data = Video_dataset(
                config.data.val_root, config.data.val_list, config.data.label_list,
                random_shift=False, num_segments=config.data.num_segments,
                modality=config.data.modality,
                image_tmpl=config.data.image_tmpl,
                transform=transform_val, test_mode=True, dense_sample=config.data.dense)            
        else:
            # 创建训练数据集和验证数据集
            from datasets.video import Video_dataset
            train_data = Video_dataset(
                config.data.train_root, config.data.train_list,
                config.data.label_list, num_segments=config.data.num_segments,
                modality=config.data.modality,
                image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
                transform=transform_train, dense_sample=config.data.dense)
            val_data = Video_dataset(
                config.data.val_root, config.data.val_list, config.data.label_list,
                random_shift=False, num_segments=config.data.num_segments,
                modality=config.data.modality,
                image_tmpl=config.data.image_tmpl,
                transform=transform_val, dense_sample=config.data.dense)   
    elif config.data.modality in ['iframe', 'mv', 'residual']:
        from datasets.compress_3 import Video_compress_dataset
        train_data = Video_compress_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense, accumulate=(not args.no_accumulation), GOP_SIZE = config.data.GOP_SIZE)
        val_data = Video_compress_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense, accumulate=(not args.no_accumulation))   

    ################ Few shot data for training ###########
    if config.data.shot:
        cls_dict = {}
        for item  in train_data.video_list:
            if item.label not in cls_dict:
                cls_dict[item.label] = [item]
            else:
                cls_dict[item.label].append(item)
        import random
        select_vids = []
        K = config.data.shot
        for category, v in cls_dict.items():
            slice = random.sample(v, K)
            select_vids.extend(slice)
        n_repeat = len(train_data.video_list) // len(select_vids)
        train_data.video_list = select_vids * n_repeat
        # print('########### number of videos: {} #########'.format(len(select_vids)))
    ########################################################


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False)

    loss_type = config.solver.loss_type
    if loss_type == 'NCE':
        criterion = NCELoss()
    elif loss_type == 'DS':
        criterion = DualLoss()
    else:
        raise NotImplementedError

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading pretrain checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], False)
            video_head.load_state_dict(checkpoint['fusion_model_state_dict'], False)
            mv_head.load_state_dict(checkpoint['fusion_model_state_dict'], False)
            del checkpoint
        else:
            logger.info("=> no pretrain checkpoint found at '{}'".format(config.resume))
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading resume checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            # mv_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded resume checkpoint '{}' (epoch {})"
                   .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no resume checkpoint found at '{}'".format(config.pretrain))

    classes,n_class = text_prompt(train_data, config)    # torch.Size([51, 77])    使用vita的时候，返回的是类别名


    if config.network.fix_text:
        for name, param in model.named_parameters():
            if "visual" not in name and "logit_scale" not in name:
                param.requires_grad_(False)
  
    if config.network.fix_video:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)

    ## freeze some parameters
    for name, param in model.named_parameters():
        if 'Adapter' in name:
            param.requires_grad = True

    # 查看可训练的参数
    # for name, param in model.named_parameters():
    #     logger.info('{}: {}'.format(name, param.requires_grad))

    optimizer = _optimizer(config, model, video_head, mv_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        video_prompt = DistributedDataParallel(video_prompt.cuda(), device_ids=[args.gpu], find_unused_parameters=True)

        if config.network.sim_header == "None" and config.network.interaction in ['DP', 'VCS']:
            video_head_nomodule = video_head
            mv_head_nomodule = mv_head
        else:
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
            mv_head = DistributedDataParallel(mv_head.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
            video_head_nomodule = video_head.module
            mv_head_nomodule = mv_head
        

    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))

        if config.data.dataset == 'charades':
            prec1, output_list, labels_list = validate_mAP(
                start_epoch,
                val_loader, classes, device,
                model, video_head, mv_head, config, n_class, logger)
        else:
            prec1, output_list, labels_list = validate(
                start_epoch,
                val_loader, classes, device,
                model, video_head, mv_head, config, n_class, logger)
        return

    #############
    save_score = True if config.data.select_topk_attributes else False
    #############

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        

        # print(model)
        train(model, video_head, mv_head, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, classes, logger, video_prompt)

        if (epoch+1) % config.logging.eval_freq == 0:
            if config.data.dataset == 'charades':
                prec1, output_list, labels_list = validate_mAP(epoch, val_loader, classes, device, model, video_head, mv_head, config, n_class, logger)
            else:
                prec1, output_list, labels_list = validate(epoch, val_loader, classes, device, model, video_head, mv_head, config, n_class, logger, save_score)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model.module, video_head_nomodule, mv_head_nomodule, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model.module, video_head_nomodule, mv_head_nomodule, optimizer)
                    if save_score:
                        save_sims(output_list, labels_list)



def train(model, video_head, mv_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger, video_prompt):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    model.train()
    video_head.train()
    mv_head.train()
    video_prompt.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    first_iteration = True
    for i,(images, mvs, residuals,list_id) in enumerate(train_loader):
        # print(list_id)     # list_id={12，45，78}  数字代表类别，个数是batchsize  
        # image.size() torch.Size([1, 16, 3, 224, 224])   b t c h w 
        # exit()
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        # lr_scheduler.step()

        data_time.update(time.time() - end)
        # b t3 h w
        images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])  # b t 3 h w
        mvs = mvs.view((-1, config.data.num_segments, 2)+ mvs.size()[-2:])  # Adjust if necessary
        residuals = residuals.view((-1, config.data.num_segments, 3) + residuals.size()[-2:]) # Adjust if necessary
        b, t, c_i, h, w = images.size()
        b, t, c_m, h, w = mvs.size()
        images = images.view(-1, c_i, h, w)  # Flatten batch and time steps
        mvs = mvs.view(-1, c_m, h, w)  # Flatten mvs similarly
        residuals = residuals.view(-1, c_i, h, w)  # Flatten residuals similarly

        texts = classes # n_cls 77

        with autocast():
            if config.solver.loss_type in ['NCE', 'DS']:
                texts = texts[list_id]  # bs 77    # torch.Size([2, 77])   [batch_size, 77]
                image_embedding, mv_embedding, cls_embedding, text_embedding, logit_scale = model(images, mvs, residuals, texts, return_token=True)
                # exit()
                # image_embedding.shape== torch.Size([32, 768])
                # cls_embedding.shape== torch.Size([2, 768])
                # text_embedding.shape== torch.Size([2, 77, 768])
                # logit_scale== tensor(95.5525, device='cuda:0', grad_fn=<ExpBackward>)
                # image_embedding.view.shape== torch.Size([2, 16, 768])
                image_embedding = image_embedding.view(b,t,-1)
                mv_embedding = mv_embedding.view(b,t,-1)
                # gather
                image_embedding = allgather(image_embedding)
                mv_embedding = allgather(mv_embedding)
                if text_embedding is not None:
                    text_embedding = allgather(text_embedding)
                cls_embedding = allgather(cls_embedding)     
                if config.network.video_prompt:
                    cls_embedding = cls_embedding.unsqueeze(1) 
                    cls_embedding = cls_embedding + video_prompt(cls_embedding, image_embedding)
                    cls_embedding = cls_embedding.squeeze()
                    print("cls_embedding-video_prompt")
                
                logits = logit_scale * video_head(image_embedding, text_embedding, cls_embedding)
                # print("The shape of logits:", logits.shape)  # 打印 logits 的形状
                # print("The content of logits:", logits)  # 打印 logits 的内容

                logits_mv = logit_scale * mv_head(mv_embedding, text_embedding, cls_embedding)
                # print("The shape of logits_mv:", logits_mv.shape)  # 打印 logits_mv 的形状
                # print("The content of logits_mv:", logits_mv)  # 打印 logits_mv 的内容
                weight_logits = 0.8
                weight_logits_mv = 0.2
                weighted_logits = logits * weight_logits
                weighted_logits_mv = logits_mv * weight_logits_mv
                combined_logits = weighted_logits + weighted_logits_mv  # 结合加权后的 logits 和 logits_mv
                # print("The shape of combined_logits:", combined_logits.shape)  # 打印 logits_mv 的形状
                # print("The content of combined_logits:", combined_logits)  # 打印 logits_mv 的内容
                
                list_id = gather_labels(list_id.to(device))  # bs -> n_gpu * bs

                ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
                # gt = [bs bs]
                # print("The shape of ground_truth:", ground_truth.shape)  # 打印 logits_mv 的形状
                # print("The content of ground_truth:", ground_truth)  # 打印 logits_mv 的内容

                loss_imgs = criterion(combined_logits, ground_truth)
                loss_texts = criterion(combined_logits.T, ground_truth)
                loss = (loss_imgs + loss_texts)/2
            else:
                raise NotImplementedError

            # loss regularization
            loss = loss / config.solver.grad_accumulation_steps

        if scaler is not None:   # 混合精度使用，报错，修改Persian参数值可以修改使用的精度
            # back propagation
            # scaler.scale(loss).backward()
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # reset gradient
        else:
            # back propagation
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()  # update param
                optimizer.zero_grad()  # reset gradient

        losses.update(loss.item(), logits.size(0))
        if first_iteration:
            # 查看使用的参数
            for name, param in model.named_parameters():
                if hasattr(param, 'grad') and param.grad is not None:
                    print(f" model Used parameter: {name}")
            # # 查看使用的参数
            # for name, param in video_head.named_parameters():
            #     if hasattr(param, 'grad') and param.grad is not None:
            #         print(f"video_head Used parameter: {name}")
            # # 查看使用的参数
            # for name, param in mv_head.named_parameters():
            #     if hasattr(param, 'grad') and param.grad is not None:
            #         print(f"mv_head Used parameter: {name}")
            first_iteration = False  # 第一次迭代结束后，将标志变量设为 False

        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))

def train_data_p(model, video_head, mv_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    model.train()
    video_head.train()
    mv_head.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()

    for i,(images, list_id) in enumerate(train_loader):
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        # lr_scheduler.step()

        data_time.update(time.time() - end)
        # b t3 h w
        if images.shape[2] == 2:
            images = images.view((-1,config.data.num_segments,2)+images.size()[-2:])  # bt 3 h w
        else:
            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])  # bt 3 h w
 
        # images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])  # bt 3 h w
        b,t,c,h,w = images.size()

        images= images.view(-1,c,h,w) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class

        texts = classes # n_cls 77





def validate(epoch, val_loader, classes, device, model, video_head, mv_head, config, n_class, logger, return_sim=False):
    top1 = AverageMeter()
    top5 = AverageMeter()
    sims_list = []
    labels_list = []
    model.eval()
    video_head.eval()
    mv_head.eval()
    with torch.no_grad():
        text_inputs = classes.to(device)  # [n_cls, 77]
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)  # [n_cls, feat_dim]
        for i,(image, mv, residual, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])  # b t 3 h w
            mv = mv.view((-1, config.data.num_segments, 2)+ mv.size()[-2:])  # Adjust if necessary
            residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:]) # Adjust if necessary
            b, t, c_i, h, w = image.size()
            b, t, c_m, h, w = mv.size()

            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c_i, h, w)
            mv_input = mv.to(device).view(-1, c_m, h, w)
            residual_input = residual.to(device).view(-1, c_i, h, w)
            image_features, mv_features, res_features = model.module.encode_image(image_input, mv_input, residual_input)
            weights = F.softmax(model.module.beta, dim=0)  # 计算权重，确保数值范围正常
            # 按权重加和特征
            merged_feats = weights[0] * image_features + weights[1] * res_features
            merged_feats = merged_feats.view(b, t, -1)
            mv_features = mv_features.view(b, t, -1)
            similarity = video_head(merged_feats, text_features, cls_feature)
            similarity_mv = mv_head(mv_features, text_features, cls_feature)

            combined_similarity = 0.8 * similarity + 0.2 * similarity_mv
            final_similarity = combined_similarity
            final_similarity = final_similarity.view(b, -1, n_class).softmax(dim=-1)  # [bs, n_frames, n_cls]
            final_similarity = final_similarity.mean(dim=1, keepdim=False)  # [bs, n_cls]

            # similarity = similarity.view(b, -1, n_class).softmax(dim=-1)  # [bs, n_frames, n_cls]
            # similarity = similarity.mean(dim=1, keepdim=False)  # [bs, n_cls]
            # similarity_mv = similarity_mv.view(b, -1, n_class).softmax(dim=-1)  # [bs, n_frames, n_cls]
            # similarity_mv = similarity_mv.mean(dim=1, keepdim=False)  # [bs, n_cls]
            if return_sim:
                sims = allgather(final_similarity)
                labels = gather_labels(class_id)
                sims_list.append(sims)
                labels_list.append(labels)

            prec = accuracy(final_similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    if return_sim:
        return top1.avg, sims_list, labels_list
    else:
        return top1.avg, None, None

def validate_mAP(epoch, val_loader, classes, device, model, video_head, mv_head, config, n_class, logger):
    mAP = AverageMeter()
    model.eval()
    video_head.eval()
    mv_head.eval()

    from torchnet import meter
    maper = meter.mAPMeter()
    sims_list = []
    labels_list = []

    with torch.no_grad():
        text_inputs = classes.to(device)  # [400, 77]
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)  # [400, 512]
        
        for i, (image, mv, residual, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])  # bt 3 h w
            # mv 和 residual 也做类似处理
            mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])  # bt 2 h w
            residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])  # bt 3 h w

            b, t, c_i, h, w = image.size()
            b, t, c_m, h, w = mv.size()
            class_id = class_id.to(device)

            image_input = image.to(device).view(-1, c_i, h, w)
            mv_input = mv.to(device).view(-1, c_m, h, w)  # Assuming 2 channels for mv
            residual_input = residual.to(device).view(-1, 3, h, w)

            # 计算每种输入的特征
            image_features = model.module.encode_image(image_input).view(b, t, -1)
            mv_features = model.module.encode_image(mv_input).view(b, t, -1)
            residual_features = model.module.encode_image(residual_input).view(b, t, -1)
            weights = F.softmax(model.module.beta, dim=0)
            merged_features = weights[0]* image_features + weights[1]* residual_features
            # 使用video_head计算相似度
            video_similarity = video_head(merged_features, text_features, cls_feature)
            mv_similarity = mv_head(mv_features, text_features, cls_feature)

            # 融合两种相似度 (可以根据需求选择不同的融合方式)
            combined_similarity = 0.8 * video_similarity + 0.2 * mv_similarity  # 加权平均，如果有不同的权重需求，可以调整

            # 处理相似度并计算mAP
            combined_similarity = combined_similarity.view(b, -1, n_class).softmax(dim=-1)  # [bs, 16, 400]
            combined_similarity = combined_similarity.mean(dim=1, keepdim=False)  # [bs, 400]
            combined_similarity = F.softmax(combined_similarity, dim=1)  # [bs, 400]

            output = allgather(combined_similarity)  # gather multi-GPU
            labels = gather_labels(class_id)  # gather class labels
            sims_list.append(output)
            labels_list.append(labels)

            maper.add(output, labels)
            mAP.update(maper.value().numpy(), labels.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}], mAP: {map:.3f}%\t'.format(i, len(val_loader), map=mAP.avg * 100)))

    logger.info(('Testing Results mAP === {mAP_result:.3f}%'.format(mAP_result=mAP.avg * 100)))
    return mAP.avg * 100, sims_list, labels_list


def save_sims(output_list, labels_list):
    outputs_sim = torch.cat(output_list, dim=0)
    labels_list_res = torch.cat(labels_list, dim=0)
    prec = accuracy(outputs_sim, labels_list_res, topk=(1, 5))
    torch.save(outputs_sim, 'video_sentence_fusion/k400_video_sims.pt')
    torch.save(labels_list_res, 'video_sentence_fusion/k400_video_labels.pt')
    # print('outputs_sim.shape==', outputs_sim.shape)
    # print('labels_list_res.shape===', labels_list_res.shape)
    # print('top1====', prec[0].item())

if __name__ == '__main__':
    args = get_parser() 
    main(args)

