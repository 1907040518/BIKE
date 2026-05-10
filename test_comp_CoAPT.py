#!/usr/bin/env python3
"""Evaluation script aligned with train_comp_CoAPT.py for compressed modalities."""
import argparse
import datetime
import json
import os
import pprint
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import clip
import torchvision
import yaml
from dotmap import DotMap

from datasets.transforms import (GroupCenterCrop, GroupFullResSample, GroupNormalize,
                                 GroupOverSample, GroupScale, Stack,
                                 ToTorchFormatTensor)

from modules.text_prompt import text_prompt
from modules.video_clip import video_header
from utils.logger import setup_logger
from utils.utils import AverageMeter, init_distributed_mode, reduce_tensor

try:
    from Coviar.transforms import get_compress_augmentation
except ImportError:
    get_compress_augmentation = None


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config yaml")
    parser.add_argument("--weights", type=str, required=True, help="model checkpoint")
    parser.add_argument("--test-list", type=str, default=None, help="override test list path")
    parser.add_argument("--dist_url", default="env://", help="dist init url")
    parser.add_argument("--world_size", default=1, type=int, help="number of processes")
    parser.add_argument("--local_rank", type=int, help="local rank for DDP")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp32")
    parser.add_argument("--test_crops", type=int, default=1)
    parser.add_argument("--test_clips", type=int, default=1)
    parser.add_argument("--dense", action="store_true", help="use dense sampling for RGB modalities")
    parser.add_argument("--no-accumulation", action="store_true", help="disable residual accumulation for compressed data")
    # ✅ 新增：保存每个样本预测结果的路径
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="path to save per-sample predictions (txt file)")
    return parser


class CoAPTBiasAdapter(nn.Module):
    """Meta-network matching train_comp_CoAPT for applying CoAPT bias."""

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

        if text_features.dim() == 2 and text_features.size(0) == video_features.size(0):
            video_features = self._normalize(video_features)
            text_features = self._normalize(text_features)
            fusion = torch.cat([text_features, video_features], dim=-1)
            bias = self.meta_net(fusion)
            return text_features + bias

        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0).expand(video_features.size(0), -1, -1)
        elif text_features.dim() == 3:
            if text_features.size(0) != video_features.size(0):
                raise ValueError("Text/video batch mismatch for CoAPT bias")
        else:
            raise ValueError("Unsupported text feature shape for CoAPT bias")

        video_features = video_features.unsqueeze(1).expand_as(text_features)
        video_features = self._normalize(video_features)
        text_features = self._normalize(text_features)

        fusion = torch.cat([text_features, video_features], dim=-1)
        fusion = fusion.view(-1, fusion.size(-1))
        bias = self.meta_net(fusion).view_as(text_features)
        return text_features + bias


def update_dict(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _default_vocab_root():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "CoAPT-main" / "VOCAB" / "gpt-L"


def load_attribute_descriptions(classnames, attribute_cfg, logger, dataset_name):
    vocab_root = attribute_cfg.get("vocab_root")
    if vocab_root is None:
        vocab_root = _default_vocab_root()
    vocab_root = Path(vocab_root)

    dataset_key = attribute_cfg.get("dataset_key", dataset_name)
    if not isinstance(dataset_key, str):
        dataset_key = str(dataset_key)
    dataset_key = dataset_key.replace("-", "_")
    dataset_key_upper = dataset_key.upper()

    seed_index = attribute_cfg.get("seed_index", attribute_cfg.get("seed", 1))
    if isinstance(seed_index, (list, tuple)) and seed_index:
        seed_index = seed_index[0]
    seed_index = int(seed_index)

    vocab_file = attribute_cfg.get("vocab_file")
    if vocab_file is None:
        vocab_file = f"{dataset_key_upper}_{seed_index}.json"
    vocab_path = vocab_root / vocab_file
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Attribute vocab file not found: {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    num_attributes = int(attribute_cfg.get("num_attributes", attribute_cfg.get("num_attr", 16)))
    attribute_words = []
    missing_classes = []
    for name in classnames:
        candidates = [name, name.replace("_", " "), name.strip(), name.strip().title()]
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

    template = attribute_cfg.get("template", "a video about {}.")
    attr_template = attribute_cfg.get("attribute_template", None)
    finalize_with_period = attribute_cfg.get("append_period", True)

    prompts = []
    for name, attr_text in zip(classnames, attribute_texts):
        attr_text = attr_text.strip().rstrip(".")

        if template.count("{}") >= 2:
            if attr_text:
                prompt = template.format(name, attr_text)
            else:
                prompt = template.format(name, "")
        else:
            base_prompt = template.format(name)
            if attr_template is not None and attr_template.count("{}") >= 1 and attr_text:
                formatted_attr = attr_template.format(attr_text)
            else:
                formatted_attr = attr_text

            prompt = base_prompt
            if formatted_attr:
                prompt = f"{prompt} {formatted_attr}".strip()

        if finalize_with_period and prompt and prompt[-1] not in {".", "!", "?"}:
            prompt = f"{prompt}."

        prompts.append(prompt)

    tokenized = torch.cat([clip.tokenize(p) for p in prompts])
    return tokenized, prompts


def build_test_dataset(config, args):
    dense_sample = getattr(config.data, "dense", False) or args.dense
    test_list = getattr(config.data, "test_list", None)
    if args.test_list is not None:
        test_list = args.test_list
    if not test_list:
        test_list = config.data.val_list

    if config.data.modality in ["mv", "residual", "iframe"]:
        if get_compress_augmentation is None:
            raise RuntimeError("Coviar transforms are required for compressed modalities")
        from datasets.compress_3 import Video_compress_dataset
        transform = get_compress_augmentation(False, config)
        dataset = Video_compress_dataset(
            config.data.val_root,
            test_list,
            config.data.label_list,
            num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            random_shift=False,
            transform=transform,
            dense_sample=dense_sample,
            test_mode=True,
            accumulate=(not args.no_accumulation),
            GOP_SIZE=config.data.GOP_SIZE,
        )
        return dataset

    from datasets.video import Video_dataset
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    if "something" in config.data.dataset:
        scale_size = (240, 320)
    else:
        scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(config.data.input_size),
        ])
    elif args.test_crops == 3:
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=config.data.input_size,
                scale_size=scale_size,
                flip=False,
            )
        ])
    elif args.test_crops == 5:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=config.data.input_size,
                scale_size=scale_size,
                flip=False,
            )
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=config.data.input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError(f"Unsupported number of test crops: {args.test_crops}")

    dataset = Video_dataset(
        config.data.val_root,
        test_list,
        config.data.label_list,
        random_shift=False,
        num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]),
        dense_sample=dense_sample,
        test_mode=True,
        test_clips=args.test_clips,
    )
    return dataset


def configure_attribute_modules(model, config, logger, attribute_prompt_cfg, attribute_prompt_enabled):
    coapt_bias_cfg = config.network.get("coapt_bias", None)
    if coapt_bias_cfg is None and isinstance(attribute_prompt_cfg, dict):
        coapt_bias_cfg = attribute_prompt_cfg.get("coapt_bias", attribute_prompt_cfg.get("coapt", None))

    coapt_bias_enabled = False
    coapt_hidden_layers = None
    if isinstance(coapt_bias_cfg, dict):
        coapt_bias_enabled = bool(coapt_bias_cfg.get("enable", True))
        layers_cfg = coapt_bias_cfg.get("layers", None)
        if layers_cfg is None:
            layer_keys = ["layer1", "layer2", "layer3"]
            layer_vals = [coapt_bias_cfg.get(key) for key in layer_keys if coapt_bias_cfg.get(key) is not None]
            if layer_vals:
                layers_cfg = layer_vals
        if layers_cfg is not None:
            if isinstance(layers_cfg, (list, tuple)):
                coapt_hidden_layers = list(layers_cfg)
            elif isinstance(layers_cfg, int):
                coapt_hidden_layers = [layers_cfg]
    else:
        coapt_bias_enabled = bool(coapt_bias_cfg)

    if coapt_bias_enabled:
        embed_dim = model.text_projection.shape[1]
        model.coapt_bias = CoAPTBiasAdapter(embed_dim, coapt_hidden_layers)
        if logger is not None and dist.get_rank() == 0:
            logger.info("CoAPT bias meta-net enabled for evaluation")
            if coapt_hidden_layers:
                logger.info(f"CoAPT hidden layers: {coapt_hidden_layers}")
    else:
        model.coapt_bias = None

    attribute_fusion_cfg = config.network.get("attribute_guided_fusion", None)
    attribute_fusion_enabled = False
    fusion_kwargs = {}

    if isinstance(attribute_fusion_cfg, dict):
        attribute_fusion_enabled = bool(attribute_fusion_cfg.get("enable", attribute_prompt_enabled))
        allowed_keys = {"hidden_dim", "num_heads", "dropout", "detach_text"}
        fusion_kwargs = {k: attribute_fusion_cfg[k] for k in allowed_keys if k in attribute_fusion_cfg}
    elif attribute_fusion_cfg is not None:
        attribute_fusion_enabled = bool(attribute_fusion_cfg)
    else:
        attribute_fusion_enabled = attribute_prompt_enabled

    if attribute_fusion_enabled and not attribute_prompt_enabled and logger is not None and dist.get_rank() == 0:
        logger.warning("Attribute-guided fusion requested but attribute prompts disabled; falling back to base fusion.")
        attribute_fusion_enabled = False

    model.configure_attribute_guided_fusion(enable=attribute_fusion_enabled, **fusion_kwargs)
    if logger is not None and dist.get_rank() == 0:
        logger.info(f"Attribute-guided fusion {'enabled' if attribute_fusion_enabled else 'disabled'}")
        if attribute_fusion_enabled and fusion_kwargs:
            logger.info(f"Fusion kwargs: {fusion_kwargs}")

    return attribute_fusion_enabled, coapt_bias_enabled


def validate(val_loader, classes, device, model, video_head, config, n_class, logger,
             coapt_bias_enabled=False, attribute_prompt_enabled=False,
             classnames=None, save_path=None):  # ✅ 新增 classnames 和 save_path 参数
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model

    # ✅ 用于收集每个样本的预测记录
    all_records = []

    # ✅ 获取 dataset 的样本路径列表（支持 .video_list 或 .samples）
    dataset = val_loader.dataset
    sample_paths = None
    if hasattr(dataset, "video_list"):
        # 常见格式：dataset.video_list 是 list of (path, label) 或 list of namedtuple
        raw_list = dataset.video_list
        try:
            sample_paths = [str(item.path) if hasattr(item, "path") else str(item[0]) for item in raw_list]
        except Exception:
            sample_paths = [str(item) for item in raw_list]
    elif hasattr(dataset, "samples"):
        sample_paths = [str(item[0]) for item in dataset.samples]
    elif hasattr(dataset, "data_list"):
        sample_paths = [str(item[0]) for item in dataset.data_list]

    with torch.no_grad():
        if classes is None:
            raise RuntimeError("Text classes tensor is required for evaluation")
        text_inputs = classes.to(device)
        base_cls_feature, text_features = clip_model.encode_text(text_inputs, return_token=True)
        coapt_module = clip_model.coapt_bias if (coapt_bias_enabled and hasattr(clip_model, "coapt_bias")) else None

        # ✅ 用于追踪全局样本索引（分布式场景下每个 rank 处理不同子集）
        global_sample_idx = 0

        for i, batch in enumerate(val_loader):
            if config.data.modality in ["mv", "residual", "iframe"]:
                image, mv, residual, class_id = batch

                image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
                mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
                residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])

                b, t, c_i, h, w = image.size()
                _, _, c_m, _, _ = mv.size()

                class_id = class_id.to(device)
                image_input = image.to(device).view(-1, c_i, h, w)
                mv_input = mv.to(device).view(-1, c_m, h, w)
                residual_input = residual.to(device).view(-1, c_i, h, w)

                if attribute_prompt_enabled and getattr(clip_model, "attribute_guided_fusion", None) is not None:
                    prompt_inputs = classes[class_id].to(device)
                    fused_flat, modality_weights, semantic_weights, _ = clip_model(
                        image_input, residual_input, mv_input, prompt_inputs, return_token=True
                    )
                    merged_feats = fused_flat.view(b, t, -1)
                else:
                    image_features, res_features, mvs_features = clip_model.encode_image(
                        image_input, residual_input, mv_input
                    )
                    weights = F.softmax(clip_model.beta, dim=0)
                    merged_feats = weights[0] * image_features + weights[1] * res_features + weights[2] * mvs_features
                    merged_feats = merged_feats.view(b, t, -1)
            else:
                image, class_id = batch
                if image.shape[2] == 2:
                    image = image.view((-1, config.data.num_segments, 2) + image.size()[-2:])
                else:
                    image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])

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
            similarity = similarity.view(b, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)  # [b, n_class]

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            # ✅ 收集每个样本的预测信息
            if save_path is not None:
                pred_indices = similarity.argmax(dim=1).cpu().tolist()   # 预测类别索引
                gt_indices = class_id.cpu().tolist()                      # 真实类别索引

                # ✅ 获取 DataLoader 使用的 sampler 全局索引（分布式场景）
                if hasattr(val_loader.sampler, "dataset") or isinstance(
                    val_loader.sampler, torch.utils.data.distributed.DistributedSampler
                ):
                    # 分布式采样器：每个 rank 的样本是全局 indices 的子集
                    sampler_indices = list(val_loader.sampler)
                    batch_start = i * val_loader.batch_size
                    batch_global_indices = sampler_indices[batch_start: batch_start + b]
                else:
                    # 非分布式：顺序索引
                    batch_global_indices = list(range(global_sample_idx, global_sample_idx + b))

                for j in range(b):
                    global_idx = batch_global_indices[j] if j < len(batch_global_indices) else -1
                    sample_path = sample_paths[global_idx] if (sample_paths and 0 <= global_idx < len(sample_paths)) else f"sample_{global_idx}"
                    sample_name = Path(sample_path).name

                    pred_label = classnames[pred_indices[j]] if (classnames and pred_indices[j] < len(classnames)) else str(pred_indices[j])
                    gt_label   = classnames[gt_indices[j]]   if (classnames and gt_indices[j]   < len(classnames)) else str(gt_indices[j])
                    correct    = "✓" if pred_indices[j] == gt_indices[j] else "✗"

                    all_records.append({
                        "global_idx":  global_idx,
                        "sample_path": sample_path,
                        "sample_name": sample_name,
                        "pred_idx":    pred_indices[j],
                        "pred_label":  pred_label,
                        "gt_idx":      gt_indices[j],
                        "gt_label":    gt_label,
                        "correct":     correct,
                    })

                global_sample_idx += b

            if i % config.logging.print_freq == 0 and logger is not None:
                logger.info(
                    (
                        "Test: [{0}/{1}]\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
                    ).format(i, len(val_loader), top1=top1, top5=top5)
                )

    if logger is not None:
        logger.info(
            "Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    # ✅ 只在 rank 0 写入预测文件
    # ✅ 多卡 gather：把所有 rank 的记录汇总到 rank 0
    if save_path is not None:
        save_path = Path(save_path)

        if dist.is_initialized() and dist.get_world_size() > 1:
            # 将本 rank 的 records 序列化为字节，用 dist.gather_object 汇总
            gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
            dist.gather_object(all_records, gathered, dst=0)

            if dist.get_rank() == 0:
                # 展平所有 rank 的记录列表
                merged_records = []
                for rank_records in gathered:
                    merged_records.extend(rank_records)
            else:
                merged_records = []   # 非 rank 0 不写文件
        else:
            # 单卡直接用
            merged_records = all_records

        # 只有 rank 0（或单卡）执行写文件
        if dist.get_rank() == 0 or not dist.is_initialized():
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 按 global_idx 排序，保证输出顺序与原始数据集一致
            merged_records.sort(key=lambda x: x["global_idx"])

            with open(save_path, "w", encoding="utf-8") as f:
                header = f"{'global_idx':<12}{'correct':<8}{'pred_label':<40}{'gt_label':<40}{'sample_name':<50}sample_path\n"
                f.write(header)
                f.write("-" * 180 + "\n")
                for rec in merged_records:
                    f.write(
                        f"{rec['global_idx']:<12}"
                        f"{rec['correct']:<8}"
                        f"{rec['pred_label']:<40}"
                        f"{rec['gt_label']:<40}"
                        f"{rec['sample_name']:<50}"
                        f"{rec['sample_path']}\n"
                    )

            if logger is not None:
                logger.info(f"Per-sample predictions saved ({len(merged_records)} samples): {save_path}")


    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(args):
    init_distributed_mode(args)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = Path(config["data"].get("output_path", "exps")) / config["data"]["dataset"] / config["network"]["arch"]
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = working_dir / f"test_{now}"

    if dist.get_rank() == 0:
        working_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir / Path(args.config).name)

    logger = setup_logger(output=str(working_dir), distributed_rank=dist.get_rank(), name="BIKE-Test")
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info(f"- Python: {sys.version}")
    logger.info(f"- PyTorch: {torch.__version__}")
    logger.info(f"- TorchVision: {torchvision.__version__}")
    logger.info("------------------------------------")
    logger.info("Configuration:")
    logger.info(pprint.pformat(config))

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    seed = config.seed + (dist.get_rank() if dist.is_initialized() else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    residual_layers = config.network.get("residual_layers_to_use", None)
    mvs_layers = config.network.get("mvs_layers_to_use", None)

    model, clip_state_dict = clip.load(
        config.network.arch,
        device="cpu",
        jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st=config.network.joint_st,
        residual_layers_to_use=residual_layers,
        mvs_layers_to_use=mvs_layers,
    )

    transform_info = get_compress_augmentation if config.data.modality in ["mv", "residual", "iframe"] else None
    logger.info(f"Evaluation modality: {config.data.modality}")

    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
    )

    if args.precision in {"amp", "fp32"}:
        model = model.float()

    test_dataset = build_test_dataset(config, args)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = None
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=True,          # ← 新增：加速 CPU→GPU 传输
        prefetch_factor=4,        # ← 新增：提前预取 4 个 batch
        persistent_workers=True,  # ← 新增：避免每次重建 worker 进程
    )

    if hasattr(test_dataset, "classes"):
        classnames = [c for _, c in test_dataset.classes]
    else:
        raise RuntimeError("Dataset must provide class names for text prompts")

    attribute_prompt_cfg = config.network.get("action_prompt", config.network.get("attribute_prompt", None))
    using_attribute_prompts = False
    classes = None
    n_class = len(classnames)

    if isinstance(attribute_prompt_cfg, dict) and attribute_prompt_cfg.get("enable", False):
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
        except FileNotFoundError as exc:
            using_attribute_prompts = False
            if dist.get_rank() == 0:
                logger.warning(f"Attribute prompt vocab not found: {exc}; falling back to class name prompts")

    if not using_attribute_prompts:
        classes, n_class = text_prompt(test_dataset)

    classes = classes.to(device)

    _attribute_fusion_enabled, coapt_bias_enabled = configure_attribute_modules(
        model,
        config,
        logger,
        attribute_prompt_cfg,
        using_attribute_prompts,
    )

    if args.weights and os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location="cpu")
        if dist.get_rank() == 0:
            logger.info(f"Loaded checkpoint '{args.weights}' (epoch {checkpoint.get('epoch', 'N/A')})")
        model.load_state_dict(update_dict(checkpoint["model_state_dict"]), strict=False)
        video_head.load_state_dict(update_dict(checkpoint["fusion_model_state_dict"]), strict=False)
        del checkpoint
    else:
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    if args.precision == "fp16":
        model.half()

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        if config.network.sim_header != "None":
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])
    else:
        model = model.to(device)
        video_head = video_head.to(device)

    top1, top5 = validate(
        test_loader,
        classes,
        device,
        model,
        video_head,
        config,
        n_class,
        logger,
        coapt_bias_enabled=coapt_bias_enabled,
        attribute_prompt_enabled=using_attribute_prompts,
        classnames=classnames,              # ✅ 传入类别名列表
        save_path=args.save_predictions,    # ✅ 传入保存路径
    )

    if dist.get_rank() == 0:
        logger.info(f"Final Test Prec@1: {top1:.3f}")
        logger.info(f"Final Test Prec@5: {top5:.3f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
