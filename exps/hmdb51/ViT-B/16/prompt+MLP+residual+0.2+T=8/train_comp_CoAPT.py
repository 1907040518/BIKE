import os
import sys
import time
import argparse
import json

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


class CoAPTBiasAdapter(nn.Module):
    """简单的 CoAPT 元网络：结合视频与文本特征，为文本分支生成偏置。"""

    def __init__(self, embed_dim, hidden_dims=None):
        super().__init__()
        if not hidden_dims:
            hidden_dims = [embed_dim * 2, embed_dim]

        layers = []
        input_dim = embed_dim * 2
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim
        layers.append(nn.Linear(input_dim, embed_dim))
        self.meta_net = nn.Sequential(*layers)

    def _normalize(self, tensor):
        return F.normalize(tensor, dim=-1, eps=1e-6)

    def forward(self, video_features, text_features):
        if video_features.dim() == 1:
            video_features = video_features.unsqueeze(0)

        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)

        # 一对一（训练阶段，B x C）
        if text_features.dim() == 2 and text_features.size(0) == video_features.size(0):
            video_features = self._normalize(video_features)
            text_features = self._normalize(text_features)
            fusion = torch.cat([text_features, video_features], dim=-1)
            bias = self.meta_net(fusion)
            return text_features + bias

        # 按样本广播（验证阶段，需对每个类别生成偏置）
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0).expand(video_features.size(0), -1, -1)
        elif text_features.dim() == 3:
            if text_features.size(0) != video_features.size(0):
                raise ValueError("文本特征批次与视频特征批次不一致")
        else:
            raise ValueError("不支持的文本特征形状")

        video_features = video_features.unsqueeze(1).expand_as(text_features)
        video_features = self._normalize(video_features)
        text_features = self._normalize(text_features)

        fusion = torch.cat([text_features, video_features], dim=-1)
        fusion = fusion.view(-1, fusion.size(-1))
        bias = self.meta_net(fusion).view_as(text_features)
        return text_features + bias

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


def _default_vocab_root():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "CoAPT-main" / "VOCAB" / "gpt-L"


def load_attribute_descriptions(classnames, attribute_cfg, logger, dataset_name):
    vocab_root = attribute_cfg.get('vocab_root')
    if vocab_root is None:
        vocab_root = _default_vocab_root()
    vocab_root = Path(vocab_root)

    dataset_key = attribute_cfg.get('dataset_key', dataset_name)
    if not isinstance(dataset_key, str):
        dataset_key = str(dataset_key)
    dataset_key = dataset_key.replace('-', '_')
    dataset_key_upper = dataset_key.upper()

    seed_index = attribute_cfg.get('seed_index', attribute_cfg.get('seed', 1))
    if isinstance(seed_index, (list, tuple)) and seed_index:
        seed_index = seed_index[0]
    seed_index = int(seed_index)

    vocab_file = attribute_cfg.get('vocab_file')
    if vocab_file is None:
        vocab_file = f"{dataset_key_upper}_{seed_index}.json"
    vocab_path = vocab_root / vocab_file
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Attribute vocab file not found: {vocab_path}")

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    num_attributes = int(attribute_cfg.get('num_attributes', attribute_cfg.get('num_attr', 16)))
    attribute_words = []
    missing_classes = []
    for name in classnames:
        candidates = [name, name.replace('_', ' '), name.strip(), name.strip().title()]
        key = next((cand for cand in candidates if cand in vocab_data), None)
        if key is None:
            missing_classes.append(name)
            attribute_words.append("")
            continue

        raw_text = str(vocab_data[key]).strip()
        if num_attributes > 0:
            tokens = raw_text.split()
            raw_text = " ".join(tokens[:num_attributes])
        attribute_words.append(raw_text)

    if missing_classes and logger is not None:
        logger.warning(f"Missing attribute descriptions for classes: {missing_classes}")
    elif logger is not None:
        logger.info(f"Loaded attribute descriptions from {vocab_path}")

    return attribute_words


def build_attribute_prompts(classnames, attribute_cfg, logger, dataset_name):
    attribute_texts = load_attribute_descriptions(classnames, attribute_cfg, logger, dataset_name)

    template = attribute_cfg.get('template', "a video about {}.")
    attr_template = attribute_cfg.get('attribute_template', None)
    finalize_with_period = attribute_cfg.get('append_period', True)

    prompts = []
    for name, attr_text in zip(classnames, attribute_texts):
        attr_text = attr_text.strip()
        attr_text = attr_text.rstrip('.')

        if template.count('{}') >= 2:
            if attr_text:
                prompt = template.format(name, attr_text)
            else:
                prompt = template.format(name, '')
        else:
            base_prompt = template.format(name)
            if attr_template is not None and attr_template.count('{}') >= 1 and attr_text:
                formatted_attr = attr_template.format(attr_text)
            else:
                formatted_attr = attr_text

            prompt = base_prompt
            if formatted_attr:
                prompt = f"{prompt} {formatted_attr}".strip()

        if finalize_with_period and prompt and prompt[-1] not in {'.', '!', '?'}:
            prompt = f"{prompt}."

        prompts.append(prompt)

    tokenized = torch.cat([clip.tokenize(p) for p in prompts])
    return tokenized, prompts

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local-rank", type=int,
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
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st = config.network.joint_st,
        residual_layers_to_use=residual_layers,
        mvs_layers_to_use=mvs_layers) # Must set jit=False for training  ViT-B/32

    # print(model)
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
            test_mode=True,    # 测试true
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

    classnames = [c for _, c in train_data.classes]

    attribute_prompt_cfg = config.network.get('action_prompt', config.network.get('attribute_prompt', None))
    using_attribute_prompts = False
    classes = None
    n_class = len(classnames)

    if isinstance(attribute_prompt_cfg, dict) and attribute_prompt_cfg.get('enable', False):
        try:
            classes, prompt_strings = build_attribute_prompts(
                classnames,
                attribute_prompt_cfg,
                logger,
                config.data.dataset,
            )
            using_attribute_prompts = True
            if dist.get_rank() == 0:
                logger.info("Attribute prompt strings loaded from vocabulary")
                preview = attribute_prompt_cfg.get('log_preview', 3)
                if preview:
                    for idx, text in enumerate(prompt_strings[:preview]):
                        logger.info(f"Prompt[{idx}]: {text}")
        except FileNotFoundError as exc:
            using_attribute_prompts = False
            if dist.get_rank() == 0:
                logger.warning(f"Attribute prompt vocabulary missing ({exc}); falling back to default prompts")

    if not using_attribute_prompts:
        classes, n_class = text_prompt(train_data)

    attribute_prompt_enabled = using_attribute_prompts
    classes = classes.to(device)

    coapt_bias_cfg = config.network.get('coapt_bias', None)
    if coapt_bias_cfg is None and isinstance(attribute_prompt_cfg, dict):
        coapt_bias_cfg = attribute_prompt_cfg.get('coapt_bias', attribute_prompt_cfg.get('coapt', None))

    coapt_bias_enabled = False
    coapt_hidden_layers = None
    if isinstance(coapt_bias_cfg, dict):
        coapt_bias_enabled = bool(coapt_bias_cfg.get('enable', True))
        layers_cfg = coapt_bias_cfg.get('layers', None)
        if layers_cfg is None:
            layer_keys = ['layer1', 'layer2', 'layer3']
            layer_vals = [coapt_bias_cfg.get(key) for key in layer_keys if coapt_bias_cfg.get(key) is not None]
            if layer_vals:
                layers_cfg = layer_vals
        if layers_cfg is not None:
            if isinstance(layers_cfg, (list, tuple)):
                coapt_hidden_layers = [int(v) for v in layers_cfg]
            elif isinstance(layers_cfg, int):
                coapt_hidden_layers = [int(layers_cfg)]
    else:
        coapt_bias_enabled = bool(coapt_bias_cfg)

    if coapt_bias_enabled:
        embed_dim = model.text_projection.shape[1]
        model.coapt_bias = CoAPTBiasAdapter(embed_dim, coapt_hidden_layers)
        if dist.get_rank() == 0:
            logger.info("CoAPT bias meta-net enabled")
            if coapt_hidden_layers:
                logger.info(f"CoAPT bias meta-net layers: {coapt_hidden_layers}")
    else:
        model.coapt_bias = None

    attribute_fusion_cfg = config.network.get('attribute_guided_fusion', None)
    attribute_fusion_enabled = False
    fusion_kwargs = {}

    if isinstance(attribute_fusion_cfg, dict):
        attribute_fusion_enabled = bool(attribute_fusion_cfg.get('enable', attribute_prompt_enabled))
        allowed_keys = {"hidden_dim", "num_heads", "dropout", "detach_text", "residual_scale"}
        fusion_kwargs = {k: attribute_fusion_cfg[k] for k in allowed_keys if k in attribute_fusion_cfg}
    elif attribute_fusion_cfg is not None:
        attribute_fusion_enabled = bool(attribute_fusion_cfg)
    else:
        attribute_fusion_enabled = attribute_prompt_enabled

    if attribute_fusion_enabled and not attribute_prompt_enabled and dist.get_rank() == 0:
        logger.warning("Attribute-guided fusion requires attribute prompts; disabling module.")
        attribute_fusion_enabled = False

    if attribute_fusion_enabled:
        model.configure_attribute_guided_fusion(enable=True, **fusion_kwargs)
        if dist.get_rank() == 0:
            logger.info("Attribute-guided fusion enabled")
            if fusion_kwargs:
                logger.info(f"Attribute fusion config: {fusion_kwargs}")
    else:
        model.configure_attribute_guided_fusion(enable=False)


    if config.network.fix_text:
        for name, param in model.named_parameters():
            if "visual" not in name and "logit_scale" not in name and "beta" not in name:
                if coapt_bias_enabled and "coapt_bias" in name:
                    continue
                param.requires_grad_(False)
  
    if config.network.fix_video:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)

    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)

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
                model, video_head, config, n_class, logger,
                coapt_bias_enabled=coapt_bias_enabled,
                attribute_prompt_enabled=attribute_prompt_enabled)
        else:
            prec1, output_list, labels_list = validate(
                start_epoch,
                val_loader, classes, device,
                model, video_head, config, n_class, logger,
                coapt_bias_enabled=coapt_bias_enabled,
                attribute_prompt_enabled=attribute_prompt_enabled)
        return

    #############
    save_score = True if config.data.select_topk_attributes else False
    #############

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        

        # print(model)
        train(
            model,
            video_head,
            train_loader,
            optimizer,
            criterion,
            scaler,
            epoch,
            device,
            lr_scheduler,
            config,
            classes,
            logger,
            coapt_bias_enabled,
            attribute_prompt_enabled,
        )

        if (epoch+1) % config.logging.eval_freq == 0:
            if config.data.dataset == 'charades':
                prec1, output_list, labels_list = validate_mAP(
                    epoch, val_loader, classes, device, model, video_head,
                    config, n_class, logger, coapt_bias_enabled=coapt_bias_enabled,
                    attribute_prompt_enabled=attribute_prompt_enabled)
            else:
                prec1, output_list, labels_list = validate(
                    epoch, val_loader, classes, device, model, video_head,
                    config, n_class, logger, return_sim=save_score,
                    coapt_bias_enabled=coapt_bias_enabled,
                    attribute_prompt_enabled=attribute_prompt_enabled)

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
          epoch, device, lr_scheduler, config, classes, logger,
          coapt_bias_enabled=False, attribute_prompt_enabled=False):
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
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        
        data_time.update(time.time() - end)
        
        # 数据预处理代码保持不变
        images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])
        mvs = mvs.view((-1, config.data.num_segments, 2) + mvs.size()[-2:])
        residuals = residuals.view((-1, config.data.num_segments, 3) + residuals.size()[-2:])
        
        b, t, c_i, h, w = images.size()
        b, t, c_m, h, w = mvs.size()
        mvs = mvs.view(-1, c_m, h, w)
        images = images.view(-1, c_i, h, w)
        residuals = residuals.view(-1, c_i, h, w)

        with autocast():
            if config.solver.loss_type in ['NCE', 'DS']:
                if classes is None:
                    raise RuntimeError("Text classes tensor is required for training")
                indices = list_id.to(device, dtype=torch.long)
                text_inputs = classes[indices]

                image_embedding, cls_embedding, text_embedding, logit_scale = model(images, residuals, mvs, text_inputs, return_token=True)
                
                # 重塑图像特征
                image_embedding = image_embedding.view(b, t, -1)
                clip_model = model.module if hasattr(model, "module") else model
                if coapt_bias_enabled and getattr(clip_model, "coapt_bias", None) is not None:
                    video_token = image_embedding.mean(dim=1)
                    cls_embedding = clip_model.coapt_bias(video_token, cls_embedding)

                # gather操作
                image_embedding = allgather(image_embedding)
                if text_embedding is not None:
                    text_embedding = allgather(text_embedding)
                cls_embedding = allgather(cls_embedding)
                
                logits = logit_scale * video_head(image_embedding, text_embedding, cls_embedding)

                list_id = gather_labels(list_id.to(device))
                ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
                
                loss_imgs = criterion(logits, ground_truth)
                loss_texts = criterion(logits.T, ground_truth)
                loss = (loss_imgs + loss_texts)/2
            else:
                raise NotImplementedError

            loss = loss / config.solver.grad_accumulation_steps

        # 反向传播代码保持不变
        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

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
          epoch, device, lr_scheduler, config, classes, logger,
          coapt_bias_enabled=False, attribute_prompt_enabled=False):
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

        if classes is None:
            raise RuntimeError("Text classes tensor is required for training")
        indices = list_id.to(device, dtype=torch.long)
        text_inputs = classes[indices]



def validate(epoch, val_loader, classes, device, model, video_head, config, n_class, logger,
             return_sim=False, coapt_bias_enabled=False, attribute_prompt_enabled=False):
    top1 = AverageMeter()
    top5 = AverageMeter()
    sims_list = []
    labels_list = []
    
    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        clip_model = model.module if hasattr(model, "module") else model
        if classes is None:
            raise RuntimeError("Text classes tensor is required for evaluation")
        text_inputs = classes.to(device)
        base_cls_feature, text_features = clip_model.encode_text(text_inputs, return_token=True)
        coapt_module = clip_model.coapt_bias if (coapt_bias_enabled and hasattr(clip_model, "coapt_bias")) else None
        print("")
        
        for i, (image, mv, residual, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
            residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])
            
            b, t, c_i, h, w = image.size()
            b, t, c_m, h, w = mv.size()

            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c_i, h, w)
            mv_input = mv.to(device).view(-1, c_m, h, w)
            residual_input = residual.to(device).view(-1, c_i, h, w)

            if attribute_prompt_enabled and getattr(clip_model, "attribute_guided_fusion", None) is not None:
                prompt_inputs = classes[class_id].to(device)
                
                # 🔥 关键修改: 返回值从4个变量改名
                fused_flat, modality_weights, semantic_weights, _ = clip_model(
                    image_input, residual_input, mv_input, prompt_inputs, return_token=True
                )
                # fused_flat: [B*T, D] - 融合后的特征
                # modality_weights: [B, 3] - 模态级别权重
                # semantic_weights: [B, L, 3] - 属性级别语义权重 (新版)
                # _: 其他返回值 (如果有)
                
                merged_feats = fused_flat.view(b, t, -1)
            else:
                image_features, res_features, mvs_features = clip_model.encode_image(
                    image_input, residual_input, mv_input
                )
                weights = F.softmax(clip_model.beta, dim=0)
                merged_feats = weights[0] * image_features + weights[1] * res_features + weights[2] * mvs_features
                merged_feats = merged_feats.view(b, t, -1)

            cls_feature = base_cls_feature
            if coapt_module is not None:
                video_token = merged_feats.mean(dim=1)
                cls_feature = coapt_module(video_token, base_cls_feature)

            similarity = video_head(merged_feats, text_features, cls_feature)

            similarity = similarity.view(b, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)

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

def validate_mAP(epoch, val_loader, classes, device, model, video_head, config, n_class, logger,
                 coapt_bias_enabled=False, attribute_prompt_enabled=False):
    mAP = AverageMeter()
    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model
    from torchnet import meter
    maper = meter.mAPMeter()
    sims_list = []
    labels_list = []

    with torch.no_grad():
        clip_model = model.module if hasattr(model, "module") else model
        if classes is None:
            raise RuntimeError("Text classes tensor is required for evaluation")
        text_inputs = classes.to(device)
        base_cls_feature, text_features = clip_model.encode_text(text_inputs, return_token=True)
        coapt_module = clip_model.coapt_bias if (coapt_bias_enabled and hasattr(clip_model, "coapt_bias")) else None
        for i, (image, class_id) in enumerate(val_loader):
            if image.shape[2] == 2:
                image = image.view((-1,config.data.num_segments,2)+image.size()[-2:])  # bt 2 h w
            else:
                image = image.view((-1,config.data.num_segments,3)+image.size()[-2:])  # bt 3 h w

            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)

            if attribute_prompt_enabled and getattr(clip_model, "attribute_guided_fusion", None) is not None:
                prompt_inputs = classes[class_id].to(device)
                fused_flat, _, _, _ = clip_model(image_input, None, None, prompt_inputs, return_token=True)
                merged_feats = fused_flat.view(b, t, -1)
            else:
                image_features, res_features, mvs_features = clip_model.encode_image(image_input)
                weights = F.softmax(clip_model.beta, dim=0)
                merged_feats = weights[0] * image_features + weights[1] * res_features + weights[2] * mvs_features
                merged_feats = merged_feats.view(b, t, -1)

            cls_feature = base_cls_feature
            if coapt_module is not None:
                video_token = merged_feats.mean(dim=1)
                cls_feature = coapt_module(video_token, base_cls_feature)

            similarity = video_head(merged_feats, text_features, cls_feature)

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

