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
        shutil.copy('clip/model.py', working_dir)
        shutil.copy('datasets/compress_3.py', working_dir)



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

    # 检查配置中是否存在residual_layers_to_use参数
    residual_layers = config.network.get('residual_layers_to_use', None)  # 如果不存在则为None
    mvs_layers = config.network.get('mvs_layers_to_use', None)  # 如果不存在则为None   
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
        joint_st = config.network.joint_st,
        residual_layers_to_use=residual_layers,
        mvs_layers_to_use=mvs_layers) # Must set jit=False for training  ViT-B/32
    
        # 在train.py中

    # 假设您已经加载了CLIP模型
    # model = CLIP(...) 
    # model.load_state_dict(torch.load('pretrained_clip.pth'))

    # 获取VisualTransformer的配置参数
    # vision_width = model.visual.width if hasattr(model.visual, 'width') else model.visual.conv1.weight.shape[0]
    # patch_size = model.visual.conv1.weight.shape[2]  # 假设是方形patch
    # input_resolution = model.visual.input_resolution

    # # 创建两个PatchEmbedding实例，分别用于2通道和3通道输入
    # patch_embed_mv = PatchEmbedding(
    #     input_resolution=input_resolution,
    #     patch_size=patch_size,
    #     width=vision_width,
    #     in_channels=2,
    #     emb_dropout=0.1  # 可以根据需要调整
    # )

    # patch_embed_res = PatchEmbedding(
    #     input_resolution=input_resolution,
    #     patch_size=patch_size,
    #     width=vision_width,
    #     in_channels=3,
    #     emb_dropout=0.1
    # )

    # # 从CLIP模型加载权重
    # patch_embed_mv.load_clip_visual_weights(model, mode='average')
    # patch_embed_res.load_clip_visual_weights(model)


    print(model)
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
            test_mode=True,
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
            
            # 加载主模型权重
            model.load_state_dict(checkpoint['model_state_dict'], False)
            video_head.load_state_dict(checkpoint['fusion_model_state_dict'], False)
            
            # 为ResidualEncoder和MVSEncoder加载预训练权重
            if hasattr(model, 'residual_encoder') and model.residual_encoder is not None:
                try:
                    # ResidualEncoder权重加载
                    residual_layers_to_use = getattr(config, 'residual_layers_to_use', [0, 1])
                    residual_dict = {}
                    
                    # 复制基础层
                    for key in ['conv1.weight', 'class_embedding', 'positional_embedding', 
                            'ln_pre.weight', 'ln_pre.bias', 'ln_post.weight', 'ln_post.bias', 'proj']:
                        source_key = f'visual.{key}'
                        target_key = f'residual_encoder.{key}'
                        if source_key in checkpoint['model_state_dict']:
                            residual_dict[target_key] = checkpoint['model_state_dict'][source_key]
                    
                    # 复制指定的transformer层
                    for target_idx, source_idx in enumerate(residual_layers_to_use):
                        for param in ['attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
                                    'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
                                    'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias']:
                            source_key = f'visual.transformer.resblocks.{source_idx}.{param}'
                            target_key = f'residual_encoder.transformer_blocks.{target_idx}.{param}'
                            if source_key in checkpoint['model_state_dict']:
                                residual_dict[target_key] = checkpoint['model_state_dict'][source_key]
                    
                    # 加载到模型
                    model.load_state_dict(residual_dict, strict=False)
                    logger.info(f"=> loaded pretrained weights for ResidualEncoder from layers {residual_layers_to_use}")
                    
                except Exception as e:
                    logger.warning(f"=> failed to load ResidualEncoder weights: {e}")
            
            # 为MVSEncoder加载预训练权重
            if hasattr(model, 'mvs_encoder') and model.mvs_encoder is not None:
                try:
                    # MVSEncoder权重加载
                    mvs_layers_to_use = getattr(config, 'mvs_layers_to_use', [0, 1])
                    mvs_dict = {}
                    
                    # 复制基础层
                    for key in ['conv1.weight', 'class_embedding', 'positional_embedding', 
                            'ln_pre.weight', 'ln_pre.bias', 'ln_post.weight', 'ln_post.bias', 'proj']:
                        source_key = f'visual.{key}'
                        target_key = f'mvs_encoder.{key}'
                        if source_key in checkpoint['model_state_dict']:
                            mvs_dict[target_key] = checkpoint['model_state_dict'][source_key]
                    
                    # 复制指定的transformer层
                    for target_idx, source_idx in enumerate(mvs_layers_to_use):
                        for param in ['attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
                                    'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
                                    'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias']:
                            source_key = f'visual.transformer.resblocks.{source_idx}.{param}'
                            target_key = f'mvs_encoder.transformer_blocks.{target_idx}.{param}'
                            if source_key in checkpoint['model_state_dict']:
                                mvs_dict[target_key] = checkpoint['model_state_dict'][source_key]
                    
                    # 加载到模型
                    model.load_state_dict(mvs_dict, strict=False)
                    logger.info(f"=> loaded pretrained weights for MVSEncoder from layers {mvs_layers_to_use}")
                    
                except Exception as e:
                    logger.warning(f"=> failed to load MVSEncoder weights: {e}")
            
            del checkpoint
        else:
            logger.info("=> no pretrain checkpoint found at '{}'".format(config.pretrain))



    classes,n_class = text_prompt(train_data, config)    # torch.Size([51, 77])    使用vita的时候，返回的是类别名


    if config.network.fix_text:
        for name, param in model.named_parameters():
            if "visual" not in name and "logit_scale" not in name and "beta" not in name:
                param.requires_grad_(False)
  
    if config.network.fix_video:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)

    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=False)

        if config.network.sim_header == "None" and config.network.interaction in ['DP', 'VCS']:
            video_head_nomodule = video_head
        else:
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
            video_head_nomodule = video_head.module
        

    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))

        if config.data.dataset == 'charades':
            prec1, output_list, labels_list = validate_mAP(
                start_epoch,
                val_loader, classes, device,
                model, video_head, config, n_class, logger)
        else:
            prec1, output_list, labels_list = validate(
                start_epoch,
                val_loader, classes, device,
                model, video_head, config, n_class, logger)
        return

    #############
    save_score = True if config.data.select_topk_attributes else False
    #############

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        

        # print(model)
        train(model, video_head, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, classes, logger)

        if (epoch+1) % config.logging.eval_freq == 0:
            if config.data.dataset == 'charades':
                prec1, output_list, labels_list = validate_mAP(epoch, val_loader, classes, device, model, video_head, config, n_class, logger)
            else:
                prec1, output_list, labels_list = validate(epoch, val_loader, classes, device, model, video_head, config, n_class, logger, save_score)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model.module, video_head_nomodule, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model.module, video_head_nomodule, optimizer)
                    if save_score:
                        save_sims(output_list, labels_list)


def train(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    model.train()
    video_head.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
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
        ## 处理MV
        mvs = mvs.view((-1, config.data.num_segments, 2) + mvs.size()[-2:])  # b t 3 h w
        # print("mvs.dataloader ",mvs.shape)
        residuals = residuals.view((-1, config.data.num_segments, 3) + residuals.size()[-2:]) # Adjust if necessary
        b, t, c_i, h, w = images.size()
        b, t, c_m, h, w = mvs.size()
        mvs = mvs.view(-1, c_m, h, w)
        images = images.view(-1, c_i, h, w)  # Flatten batch and time steps  b*t c h w 
        residuals = residuals.view(-1, c_i, h, w)  # Flatten residuals similarly
        # Embedding MV RES
        # print("mvs.shape", mvs.shape)
        # embedded_mv = patch_embed_mv(mvs)
        # print("embedded.mvs.shape", embedded_mv.shape)

        # print("res.shape", residuals.shape)
        # embedded_res = patch_embed_res(residuals)
        # print("embeddedres.shape", embedded_res.shape)

        # print("images.shape", images.shape)
        texts = classes # n_cls 77

        with autocast():
            if config.solver.loss_type in ['NCE', 'DS']:
                texts = texts[list_id]  # bs 77    # torch.Size([2, 77])   [batch_size, 77]
                image_embedding, cls_embedding, text_embedding, logit_scale = model(images, residuals, mvs, texts, return_token=True)
                # embedding ，将prompt加在image前
                # num_prompts = 3  # 添加的prompt tokens数量
                
                # # 原始形状是[b*t, feature_dim]，需要重新计算调整后的维度
                # new_feature_dim = image_embedding.size(1) // t
                # if image_embedding.size(0) != b*t:  # 如果大小不匹配
                #     # 首先获取有效的embedding形状
                #     total_tokens = image_embedding.size(0) // b
                #     image_embedding = image_embedding.view(b, total_tokens, -1)
                #     # 舍弃prompt tokens（假设放在开头）
                #     image_embedding = image_embedding[:, num_prompts:, :].contiguous()
                #     # 重新整形为[b, t, -1]
                #     image_embedding = image_embedding.view(b, t, -1)
                # else:
                #     # 正常重塑
                #     image_embedding = image_embedding.view(b, t, -1)
                # image_embedding.shape== torch.Size([32, 768])
                # cls_embedding.shape== torch.Size([2, 768])
                # text_embedding.shape== torch.Size([2, 77, 768])
                # logit_scale== tensor(95.5525, device='cuda:0', grad_fn=<ExpBackward>)
                # image_embedding.view.shape== torch.Size([2, 16, 768])
                image_embedding = image_embedding.view(b,t,-1)
                # gather
                image_embedding = allgather(image_embedding)
                if text_embedding is not None:
                    text_embedding = allgather(text_embedding)
                cls_embedding = allgather(cls_embedding)     
                logits = logit_scale * video_head(image_embedding, text_embedding, cls_embedding)

                list_id = gather_labels(list_id.to(device))  # bs -> n_gpu * bs

                ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
                # gt = [bs bs]
                loss_imgs = criterion(logits, ground_truth)
                loss_texts = criterion(logits.T, ground_truth)
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

def train_data_p(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    model.train()
    video_head.train()
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





def validate(epoch, val_loader, classes, device, model, video_head, config, n_class, logger, return_sim=False):
    top1 = AverageMeter()
    top5 = AverageMeter()
    sims_list = []
    labels_list = []
    model.eval()
    video_head.eval()

    with torch.no_grad():
        text_inputs = classes.to(device)  # [n_cls, 77]
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)  # [n_cls, feat_dim]  [cla,77,featdim512]
        for i,(image, mv, residual, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])  # b t 3 h w
            mv = mv.view((-1, config.data.num_segments, 2)+ mv.size()[-2:])  # Adjust if necessary
            residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:]) # Adjust if necessary
            b, t, c_i, h, w = image.size()
            b, t, c_m, h, w = mv.size()

            # if image.shape[2] == 2:
            #     image = image.view((-1,config.data.num_segments,2)+image.size()[-2:])  # bt 3 h w
            # else:
            #     image = image.view((-1,config.data.num_segments,3)+image.size()[-2:])  # bt 3 h w
    
            # image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            # b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c_i, h, w)

            mv_input = mv.to(device).view(-1, c_m, h, w)
            residual_input = residual.to(device).view(-1, c_i, h, w)

            image_features, res_features, mvs_features  = model.module.encode_image(image_input, residual_input, mv_input)
            weights = F.softmax(model.module.beta, dim=0)  # 计算权重，确保数值范围正常
            # 按权重加和特征
            merged_feats = weights[0] * image_features + weights[1] * res_features + weights[2] * mvs_features

            merged_feats = merged_feats.view(b, t, -1)

            similarity = video_head(merged_feats, text_features, cls_feature)

            similarity = similarity.view(b, -1, n_class).softmax(dim=-1)  # [bs, n_frames, n_cls]
            similarity = similarity.mean(dim=1, keepdim=False)  # [bs, n_cls]

            if return_sim:
                sims = allgather(similarity)
                labels = gather_labels(class_id)
                sims_list.append(sims)
                labels_list.append(labels)

            prec = accuracy(similarity, class_id, topk=(1, 5))
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

def validate_mAP(epoch, val_loader, classes, device, model, video_head, config, n_class, logger):
    mAP = AverageMeter()
    model.eval()
    video_head.eval()
    from torchnet import meter
    maper = meter.mAPMeter()
    sims_list = []
    labels_list = []

    with torch.no_grad():
        text_inputs = classes.to(device)  # [400, 77]
        cls_feature, text_features = model.module.encode_text(text_inputs, return_token=True)  # [400, 512]
        for i, (image, class_id) in enumerate(val_loader):
            if image.shape[2] == 2:
                image = image.view((-1,config.data.num_segments,2)+image.size()[-2:])  # bt 3 h w
            else:
                image = image.view((-1,config.data.num_segments,3)+image.size()[-2:])  # bt 3 h w
    
            # image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.module.encode_image(image_input).view(b, t, -1)
            similarity = video_head(image_features, text_features, cls_feature)

            similarity = similarity.view(b, -1, n_class).softmax(dim=-1)  # [bs, 16, 400]
            similarity = similarity.mean(dim=1, keepdim=False)  # [bs, 400]
            similarity = F.softmax(similarity, dim=1)
            output = allgather(similarity)
            labels = gather_labels(class_id)
            sims_list.append(output)
            labels_list.append(labels)

            maper.add(output, labels)
            mAP.update(maper.value().numpy(),labels.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1},mAP:{map:.3f}]\t'.format(i, len(val_loader), map=mAP.avg * 100)))

    logger.info(('Testing Results mAP === {mAP_result:.3f}'.format(mAP_result=mAP.avg * 100)))
    return mAP.avg * 100, sims_list, labels_list


def save_sims(output_list, labels_list):
    outputs_sim = torch.cat(output_list, dim=0)
    labels_list_res = torch.cat(labels_list, dim=0)
    prec = accuracy(outputs_sim, labels_list_res, topk=(1, 5))
    torch.save(outputs_sim, 'video_sentence_fusion/hmdb51_video_sims.pt')
    torch.save(labels_list_res, 'video_sentence_fusion/hmdb51_video_labels.pt')
    # print('outputs_sim.shape==', outputs_sim.shape)
    # print('labels_list_res.shape===', labels_list_res.shape)
    # print('top1====', prec[0].item())

if __name__ == '__main__':
    args = get_parser() 
    main(args)

