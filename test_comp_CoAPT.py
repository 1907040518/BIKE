#!/usr/bin/env python3
"""Evaluation script aligned with train_comp_CoAPT.py for compressed modalities.

使用方法：

python test_comp_CoAPT.py --config your_config.yaml --weights best_model.pt --save_predictions results/my_predictions.txt
"""
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

from modules.text_prompt import text_prompt
from modules.video_clip import video_header
from utils.logger import setup_logger
from utils.utils import (
    AverageMeter,
    init_distributed_mode,
    reduce_tensor,
    accuracy,
    gather_labels,
)
from utils.Augmentation import get_augmentation

try:
    from Coviar.transforms import get_compress_augmentation, GroupCenterCrop, GroupScale
except ImportError:
    get_compress_augmentation = None


# ============================================================
# AllGather (与训练代码一致)
# ============================================================
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


# ============================================================
# CoAPTBiasAdapter (与训练代码完全一致)
# ============================================================
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

        # 一对一
        if text_features.dim() == 2 and text_features.size(0) == video_features.size(0):
            video_features = self._normalize(video_features)
            text_features = self._normalize(text_features)
            fusion = torch.cat([text_features, video_features], dim=-1)
            bias = self.meta_net(fusion)
            return text_features + bias

        # 按样本广播
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


# ============================================================
# 工具函数
# ============================================================
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


# ============================================================
# 命令行参数
# ============================================================
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
    parser.add_argument("--no-accumulation", action="store_true",
                        help="disable residual accumulation for compressed data")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="保存每个视频预测结果的txt文件路径，例如: predictions.txt")
    return parser


# ============================================================
# 🔥 新增：带路径信息的数据集包装器
# ============================================================
class DatasetWithIndex(torch.utils.data.Dataset):
    """
    包装任意 Video_dataset，使 __getitem__ 额外返回该样本在数据集中的
    全局索引（即清单文件中的行号，从 0 开始）。

    返回格式：
        压缩模态: (image, mv, residual, class_id, dataset_index)
        RGB 模态: (image, class_id, dataset_index)
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # sample 可能是 (image, mv, res, label) 或 (image, label)
        return (*sample, idx)

    # 透传常用属性
    @property
    def classes(self):
        return self.dataset.classes

    def get_video_path(self, idx):
        """
        根据数据集索引返回对应的视频路径。
        兼容 video3.Video_dataset / video.Video_dataset 两种实现：
          - self.dataset.video_list[idx].path
          - self.dataset.video_list[idx]  (直接是字符串)
        """
        entry = self.dataset.video_list[idx]
        if hasattr(entry, "path"):
            return entry.path
        return str(entry)


# ============================================================
# 数据集构建 (与训练代码的数据集保持一致)
# ============================================================
def build_test_dataset(config, args):
    """构建测试数据集，与训练代码保持一致的数据集类和预处理。"""
    dense_sample = getattr(config.data, "dense", False) or args.dense
    test_list = getattr(config.data, "test_list", None)
    if args.test_list is not None:
        test_list = args.test_list
    if not test_list:
        test_list = config.data.val_list

    if config.data.modality in ["mv", "residual", "iframe"]:
        if get_compress_augmentation is None:
            raise RuntimeError("Coviar transforms are required for compressed modalities")
        transform = get_compress_augmentation(False, config)

        from datasets.video3 import Video_dataset
        dataset = Video_dataset(
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
        )
    else:
        from datasets.video import Video_dataset
        transform = get_augmentation(False, config)
        dataset = Video_dataset(
            config.data.val_root,
            test_list,
            config.data.label_list,
            random_shift=False,
            num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform,
            dense_sample=dense_sample,
            test_mode=True,
        )

    # 🔥 用 DatasetWithIndex 包装，使 __getitem__ 返回数据集索引
    return DatasetWithIndex(dataset)


# ============================================================
# 属性模块配置 (与训练代码一致)
# ============================================================
def configure_attribute_modules(model, config, logger, attribute_prompt_cfg, attribute_prompt_enabled):
    """配置 CoAPT bias 和 attribute guided fusion，与训练代码逻辑完全一致。"""

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
                coapt_hidden_layers = [int(v) for v in layers_cfg]
            elif isinstance(layers_cfg, int):
                coapt_hidden_layers = [int(layers_cfg)]
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

    if attribute_fusion_enabled and not attribute_prompt_enabled:
        if logger is not None and dist.get_rank() == 0:
            logger.warning("Attribute-guided fusion requires attribute prompts; disabling module.")
        attribute_fusion_enabled = False

    if attribute_fusion_enabled:
        model.configure_attribute_guided_fusion(enable=True, **fusion_kwargs)
        if logger is not None and dist.get_rank() == 0:
            logger.info("Attribute-guided fusion enabled")
            if fusion_kwargs:
                logger.info(f"Attribute fusion config: {fusion_kwargs}")
    else:
        model.configure_attribute_guided_fusion(enable=False)

    return attribute_fusion_enabled, coapt_bias_enabled


# ============================================================
# 🔥 保存每个视频预测结果的函数（支持路径 + 清单顺序）
# ============================================================
def save_video_predictions(prediction_records, classnames, save_path, logger=None):
    """
    将测试集中每个视频的预测结果保存到 txt 文件。

    prediction_records 中每条记录包含：
        - 'dataset_index' : int,  该样本在数据集（清单文件）中的行号（0-based）
        - 'video_path'    : str,  视频文件路径（来自清单文件）
        - 'ground_truth'  : int,  真实类别索引
        - 'predicted'     : int,  预测类别索引
        - 'correct'       : bool, 是否预测正确
        - 'top5_preds'    : list[int],   Top-5 类别索引
        - 'top5_scores'   : list[float], Top-5 置信度
    """
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 🔑 按清单文件行号（dataset_index）排序，保证输出顺序与清单一致
    prediction_records.sort(key=lambda x: x["dataset_index"])

    total = len(prediction_records)
    correct_count = sum(1 for r in prediction_records if r["correct"])
    wrong_count = total - correct_count
    acc = correct_count / total * 100.0 if total > 0 else 0.0

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("Video Prediction Results\n")
        f.write(f"Total: {total} | Correct: {correct_count} | Wrong: {wrong_count} | Accuracy: {acc:.2f}%\n")
        f.write("=" * 120 + "\n\n")

        # --- 全量预测列表（顺序与清单文件一致）---
        f.write("-" * 120 + "\n")
        f.write(
            f"{'ListIdx':<9} {'Correct?':<10} {'GT Label':<28} {'Pred Label':<28} "
            f"{'Confidence':<12} Video Path\n"
        )
        f.write("-" * 120 + "\n")

        for record in prediction_records:
            ds_idx     = record["dataset_index"]
            gt_idx     = record["ground_truth"]
            pred_idx   = record["predicted"]
            is_correct = record["correct"]
            top5_scores = record["top5_scores"]
            video_path  = record.get("video_path", "N/A")

            gt_name   = classnames[gt_idx]   if gt_idx   < len(classnames) else f"class_{gt_idx}"
            pred_name = classnames[pred_idx] if pred_idx < len(classnames) else f"class_{pred_idx}"
            confidence = top5_scores[0] if top5_scores else 0.0
            status = "✓" if is_correct else "✗"

            f.write(
                f"{ds_idx:<9} {status:<10} {gt_name:<28} {pred_name:<28} "
                f"{confidence:<12.4f} {video_path}\n"
            )

        f.write("\n")

        # --- 错误预测详情（含 Top-5）---
        f.write("=" * 120 + "\n")
        f.write("WRONG PREDICTIONS (Details with Top-5)\n")
        f.write("=" * 120 + "\n\n")

        wrong_records = [r for r in prediction_records if not r["correct"]]
        if not wrong_records:
            f.write("All predictions are correct!\n")
        else:
            for record in wrong_records:
                ds_idx     = record["dataset_index"]
                gt_idx     = record["ground_truth"]
                pred_idx   = record["predicted"]
                top5_preds  = record["top5_preds"]
                top5_scores = record["top5_scores"]
                video_path  = record.get("video_path", "N/A")

                gt_name   = classnames[gt_idx]   if gt_idx   < len(classnames) else f"class_{gt_idx}"
                pred_name = classnames[pred_idx] if pred_idx < len(classnames) else f"class_{pred_idx}"

                f.write(f"List Index   : {ds_idx}\n")
                f.write(f"Video Path   : {video_path}\n")
                f.write(f"Ground Truth : [{gt_idx}] {gt_name}\n")
                f.write(f"Predicted    : [{pred_idx}] {pred_name}\n")
                f.write("Top-5 Predictions:\n")
                for rank, (cls_idx, score) in enumerate(zip(top5_preds, top5_scores)):
                    cls_name = classnames[cls_idx] if cls_idx < len(classnames) else f"class_{cls_idx}"
                    marker = " <-- GT" if cls_idx == gt_idx else ""
                    f.write(f"  #{rank+1}: [{cls_idx}] {cls_name}  (score={score:.4f}){marker}\n")
                f.write("\n")

        # --- 按类别统计准确率 ---
        f.write("=" * 120 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("=" * 120 + "\n\n")

        class_correct = {}
        class_total   = {}
        for record in prediction_records:
            gt = record["ground_truth"]
            class_total[gt]   = class_total.get(gt, 0) + 1
            if record["correct"]:
                class_correct[gt] = class_correct.get(gt, 0) + 1

        f.write(f"{'Class Index':<12} {'Class Name':<35} {'Correct':<10} {'Total':<10} {'Accuracy':<10}\n")
        f.write("-" * 77 + "\n")
        for cls_idx in sorted(class_total.keys()):
            cls_name = classnames[cls_idx] if cls_idx < len(classnames) else f"class_{cls_idx}"
            correct  = class_correct.get(cls_idx, 0)
            total    = class_total[cls_idx]
            cls_acc  = correct / total * 100.0 if total > 0 else 0.0
            f.write(f"{cls_idx:<12} {cls_name:<35} {correct:<10} {total:<10} {cls_acc:<10.2f}%\n")

    if logger is not None:
        logger.info(f"Prediction results saved to: {save_path}")
        logger.info(f"  Total={total}, Correct={correct_count}, Wrong={wrong_count}, Acc={acc:.2f}%")


# ============================================================
# validate (与训练代码的 validate 完全对齐)
# ============================================================
def validate(val_loader, classes, device, model, video_head, config, n_class, logger,
             coapt_bias_enabled=False, attribute_prompt_enabled=False,
             save_predictions_path=None, classnames=None,
             test_dataset=None):          # 🔥 新增：传入数据集用于反查路径
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model

    prediction_records = []

    with torch.no_grad():
        if classes is None:
            raise RuntimeError("Text classes tensor is required for evaluation")

        text_inputs = classes.to(device)
        base_cls_feature, text_features = clip_model.encode_text(text_inputs, return_token=True)
        coapt_module = (
            clip_model.coapt_bias
            if (coapt_bias_enabled and hasattr(clip_model, "coapt_bias"))
            else None
        )

        base_weights = F.softmax(clip_model.beta, dim=0)

        for i, batch in enumerate(val_loader):
            if config.data.modality in ["mv", "residual", "iframe"]:
                # 🔥 DatasetWithIndex 使 batch 末尾多一个 dataset_indices
                image, mv, residual, class_id, dataset_indices = batch

                image    = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
                mv       = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
                residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])

                b, t, c_i, h, w = image.size()
                _, _, c_m, _, _ = mv.size()

                class_id    = class_id.to(device)
                image_input    = image.to(device).view(-1, c_i, h, w)
                mv_input       = mv.to(device).view(-1, c_m, h, w)
                residual_input = residual.to(device).view(-1, c_i, h, w)

                image_features, res_features, mvs_features = clip_model.encode_image(
                    image_input, residual_input, mv_input
                )
                image_features = image_features.view(b, t, -1)
                res_features   = res_features.view(b, t, -1)
                mvs_features   = mvs_features.view(b, t, -1)

                if attribute_prompt_enabled and getattr(clip_model, "attribute_guided_fusion", None) is not None:
                    batch_similarities = []
                    for cls_idx in range(n_class):
                        cls_text_tokens = text_features[cls_idx:cls_idx+1].expand(b, -1, -1)
                        fused_feats, final_weights, _ = clip_model.attribute_guided_fusion(
                            image_features, mvs_features, res_features,
                            cls_text_tokens, base_weights,
                        )
                        cls_cls_feat  = base_cls_feature[cls_idx:cls_idx+1]
                        cls_text_feat = text_features[cls_idx:cls_idx+1]
                        cls_cls_feat_batch = cls_cls_feat.expand(b, -1)
                        cls_sim = video_head(
                            fused_feats,
                            cls_text_feat.expand(b, -1, -1),
                            cls_cls_feat_batch,
                        )
                        if cls_sim.dim() == 3:
                            cls_sim = cls_sim.mean(dim=1)
                        elif cls_sim.dim() == 2 and cls_sim.size(1) > 1:
                            cls_sim = cls_sim.mean(dim=1, keepdim=True)
                        elif cls_sim.dim() == 1:
                            cls_sim = cls_sim.unsqueeze(1)
                        batch_similarities.append(cls_sim)
                    similarity = torch.cat(batch_similarities, dim=1)
                    similarity = F.softmax(similarity, dim=-1)
                else:
                    merged_feats = (
                        base_weights[0] * image_features
                        + base_weights[1] * res_features
                        + base_weights[2] * mvs_features
                    )
                    cls_feature = base_cls_feature
                    if coapt_module is not None:
                        video_token = merged_feats.mean(dim=1)
                        cls_feature = coapt_module(video_token, base_cls_feature)
                    similarity = video_head(merged_feats, text_features, cls_feature)
                    similarity = similarity.view(b, -1, n_class).softmax(dim=-1)
                    similarity = similarity.mean(dim=1, keepdim=False)

            else:
                # RGB / video 模态
                image, class_id, dataset_indices = batch   # 🔥

                if image.shape[2] == 2:
                    image = image.view((-1, config.data.num_segments, 2) + image.size()[-2:])
                else:
                    image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])

                b, t, c, h, w = image.size()
                class_id    = class_id.to(device)
                image_input = image.to(device).view(-1, c, h, w)

                image_features, res_features, mvs_features = clip_model.encode_image(image_input)
                image_features = image_features.view(b, t, -1)
                if res_features is not None:
                    res_features = res_features.view(b, t, -1)
                if mvs_features is not None:
                    mvs_features = mvs_features.view(b, t, -1)

                if (attribute_prompt_enabled
                        and getattr(clip_model, "attribute_guided_fusion", None) is not None
                        and res_features is not None and mvs_features is not None):
                    batch_similarities = []
                    for cls_idx in range(n_class):
                        cls_text_tokens = text_features[cls_idx:cls_idx+1].expand(b, -1, -1)
                        fused_feats, final_weights, _ = clip_model.attribute_guided_fusion(
                            image_features, mvs_features, res_features,
                            cls_text_tokens, base_weights,
                        )
                        cls_cls_feat  = base_cls_feature[cls_idx:cls_idx+1]
                        cls_text_feat = text_features[cls_idx:cls_idx+1]
                        cls_cls_feat_batch = cls_cls_feat.expand(b, -1)
                        cls_sim = video_head(fused_feats, cls_text_feat.expand(b, -1, -1), cls_cls_feat_batch)
                        if cls_sim.dim() == 3:
                            cls_sim = cls_sim.mean(dim=1)
                        elif cls_sim.dim() == 2 and cls_sim.size(1) > 1:
                            cls_sim = cls_sim.mean(dim=1, keepdim=True)
                        elif cls_sim.dim() == 1:
                            cls_sim = cls_sim.unsqueeze(1)
                        batch_similarities.append(cls_sim)
                    similarity = torch.cat(batch_similarities, dim=1)
                    similarity = F.softmax(similarity, dim=-1)
                else:
                    weights = F.softmax(clip_model.beta, dim=0)
                    if res_features is not None and mvs_features is not None:
                        merged_feats = (weights[0] * image_features
                                        + weights[1] * res_features
                                        + weights[2] * mvs_features)
                    else:
                        merged_feats = image_features
                    merged_feats = merged_feats.view(b, t, -1)

                    cls_feature = base_cls_feature
                    if coapt_module is not None:
                        video_token = merged_feats.mean(dim=1)
                        cls_feature = coapt_module(video_token, base_cls_feature)

                    similarity = video_head(merged_feats, text_features, cls_feature)
                    similarity = similarity.view(b, -1, n_class).softmax(dim=-1)
                    similarity = similarity.mean(dim=1, keepdim=False)

            # 🔥 收集预测记录（使用 dataset_indices 反查路径）
            if save_predictions_path is not None:
                _, top5_indices = similarity.topk(5, dim=1, largest=True, sorted=True)
                top5_scores_batch = torch.gather(similarity, 1, top5_indices)

                for sample_idx in range(b):
                    ds_idx    = dataset_indices[sample_idx].item()   # 清单文件行号
                    gt_label  = class_id[sample_idx].item()
                    pred_label = top5_indices[sample_idx, 0].item()
                    is_correct = (pred_label == gt_label)

                    # 🔑 通过数据集索引反查视频路径
                    video_path = "N/A"
                    if test_dataset is not None:
                        try:
                            video_path = test_dataset.get_video_path(ds_idx)
                        except Exception:
                            pass

                    record = {
                        "dataset_index": ds_idx,          # 清单文件行号（0-based）
                        "video_path":    video_path,       # 🔥 视频路径
                        "ground_truth":  gt_label,
                        "predicted":     pred_label,
                        "correct":       is_correct,
                        "top5_preds":    top5_indices[sample_idx].cpu().tolist(),
                        "top5_scores":   top5_scores_batch[sample_idx].cpu().tolist(),
                    }
                    prediction_records.append(record)

            # 计算准确率
            prec  = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0 and logger is not None:
                logger.info(
                    "Test: [{0}/{1}]\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i, len(val_loader), top1=top1, top5=top5
                    )
                )

    if logger is not None:
        logger.info(
            "Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )

    # 🔥 保存预测结果（仅 rank 0）
    if save_predictions_path is not None and dist.get_rank() == 0:
        if classnames is not None:
            save_video_predictions(prediction_records, classnames, save_predictions_path, logger)
        else:
            if logger is not None:
                logger.warning("classnames not provided; skipping prediction saving")

    return top1.avg, top5.avg


# ============================================================
# validate_mAP (Charades 多标签，同步更新)
# ============================================================
def validate_mAP(val_loader, classes, device, model, video_head, config, n_class, logger,
                 coapt_bias_enabled=False, attribute_prompt_enabled=False,
                 save_predictions_path=None, classnames=None,
                 test_dataset=None):      # 🔥 新增
    mAP_meter = AverageMeter()
    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model

    from torchnet import meter
    maper = meter.mAPMeter()
    sims_list   = []
    labels_list = []
    prediction_records = []

    with torch.no_grad():
        if classes is None:
            raise RuntimeError("Text classes tensor is required for evaluation")
        text_inputs = classes.to(device)
        base_cls_feature, text_features = clip_model.encode_text(text_inputs, return_token=True)
        coapt_module = (
            clip_model.coapt_bias
            if (coapt_bias_enabled and hasattr(clip_model, "coapt_bias"))
            else None
        )
        base_weights = F.softmax(clip_model.beta, dim=0)

        for i, (image, class_id, dataset_indices) in enumerate(val_loader):   # 🔥
            if image.shape[2] == 2:
                image = image.view((-1, config.data.num_segments, 2) + image.size()[-2:])
            else:
                image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])

            b, t, c, h, w = image.size()
            class_id    = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)

            if attribute_prompt_enabled and getattr(clip_model, "attribute_guided_fusion", None) is not None:
                prompt_inputs = classes[class_id].to(device)
                fused_flat, _, _, _ = clip_model(image_input, None, None, prompt_inputs, return_token=True)
                merged_feats = fused_flat.view(b, t, -1)
            else:
                image_features, res_features, mvs_features = clip_model.encode_image(image_input)
                weights = F.softmax(clip_model.beta, dim=0)
                if res_features is not None and mvs_features is not None:
                    merged_feats = (weights[0] * image_features
                                    + weights[1] * res_features
                                    + weights[2] * mvs_features)
                else:
                    merged_feats = image_features
                merged_feats = merged_feats.view(b, t, -1)

            cls_feature = base_cls_feature
            if coapt_module is not None:
                video_token = merged_feats.mean(dim=1)
                cls_feature = coapt_module(video_token, base_cls_feature)

            similarity = video_head(merged_feats, text_features, cls_feature)
            similarity = similarity.view(b, -1, n_class).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            similarity = F.softmax(similarity, dim=1)

            output = allgather(similarity)
            labels = gather_labels(class_id)
            sims_list.append(output)
            labels_list.append(labels)

            maper.add(output, labels)
            mAP_meter.update(maper.value().numpy(), labels.size(0))

            if i % config.logging.print_freq == 0 and logger is not None:
                logger.info(
                    "Test: [{0}/{1}, mAP:{map:.3f}]".format(
                        i, len(val_loader), map=mAP_meter.avg * 100
                    )
                )

    if logger is not None:
        logger.info("Testing Results mAP === {mAP_result:.3f}".format(mAP_result=mAP_meter.avg * 100))

    return mAP_meter.avg * 100, sims_list, labels_list


# ============================================================
# 主函数
# ============================================================
def main(args):
    init_distributed_mode(args)

    if args.distributed:
        print("[INFO] turn on distributed evaluation", flush=True)
    else:
        print("[INFO] turn off distributed evaluation", flush=True)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = (
        Path(config["data"].get("output_path", "exps"))
        / config["data"]["dataset"]
        / config["network"]["arch"]
    )
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = working_dir / f"test_{now}"

    if dist.get_rank() == 0:
        working_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir / Path(args.config).name)

    logger = setup_logger(
        output=str(working_dir), distributed_rank=dist.get_rank(), name="BIKE-Test"
    )
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info(f"- Python: {sys.version}")
    logger.info(f"- PyTorch: {torch.__version__}")
    logger.info(f"- TorchVision: {torchvision.__version__}")
    logger.info("------------------------------------")
    logger.info("Configuration:")
    logger.info(pprint.pformat(config))
    logger.info("------------------------------------")

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    seed = config.seed + (dist.get_rank() if dist.is_initialized() else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    residual_layers = config.network.get("residual_layers_to_use", None)
    mvs_layers      = config.network.get("mvs_layers_to_use", None)

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

    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
    )

    if args.precision in {"amp", "fp32"}:
        model = model.float()

    # 构建测试数据集（已被 DatasetWithIndex 包装）
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
    )

    # 获取类别名称
    if hasattr(test_dataset, "classes"):
        classnames = [c for _, c in test_dataset.classes]
    else:
        raise RuntimeError("Dataset must provide class names for text prompts")

    # 属性提示构建
    attribute_prompt_cfg = config.network.get(
        "action_prompt", config.network.get("attribute_prompt", None)
    )
    using_attribute_prompts = False
    classes = None
    n_class = len(classnames)

    if isinstance(attribute_prompt_cfg, dict) and attribute_prompt_cfg.get("enable", False):
        try:
            classes, prompt_strings = build_attribute_prompts(
                classnames, attribute_prompt_cfg, logger, config.data.dataset
            )
            using_attribute_prompts = True
            if dist.get_rank() == 0:
                logger.info("Attribute prompt strings loaded from vocabulary")
                preview = attribute_prompt_cfg.get("log_preview", 3)
                if preview:
                    for idx, text in enumerate(prompt_strings[:preview]):
                        logger.info(f"Prompt[{idx}]: {text}")
        except FileNotFoundError as exc:
            using_attribute_prompts = False
            if dist.get_rank() == 0:
                logger.warning(
                    f"Attribute prompt vocab not found: {exc}; falling back to class name prompts"
                )

    if not using_attribute_prompts:
        classes, n_class = text_prompt(test_dataset)

    classes = classes.to(device)

    attribute_fusion_enabled, coapt_bias_enabled = configure_attribute_modules(
        model, config, logger, attribute_prompt_cfg, using_attribute_prompts
    )

    # 加载检查点
    if args.weights and os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location="cpu")
        if dist.get_rank() == 0:
            logger.info(
                f"Loaded checkpoint '{args.weights}' (epoch {checkpoint.get('epoch', 'N/A')})"
            )
        model.load_state_dict(update_dict(checkpoint["model_state_dict"]), strict=False)
        video_head.load_state_dict(
            update_dict(checkpoint["fusion_model_state_dict"]), strict=False
        )
        del checkpoint
    else:
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    if args.precision == "fp16":
        model.half()

    if args.distributed:
        model = DistributedDataParallel(
            model.cuda(), device_ids=[args.gpu], find_unused_parameters=True
        )
        if config.network.sim_header != "None":
            video_head = DistributedDataParallel(
                video_head.cuda(), device_ids=[args.gpu]
            )
        else:
            video_head = video_head.cuda()
    else:
        model = model.to(device)
        video_head = video_head.to(device)

    save_predictions_path = args.save_predictions
    if save_predictions_path is None:
        save_predictions_path = str(working_dir / "video_predictions.txt")

    # 运行验证
    if config.data.dataset == "charades":
        mAP, _, _ = validate_mAP(
            test_loader, classes, device, model, video_head, config, n_class, logger,
            coapt_bias_enabled=coapt_bias_enabled,
            attribute_prompt_enabled=using_attribute_prompts,
            save_predictions_path=save_predictions_path,
            classnames=classnames,
            test_dataset=test_dataset,    # 🔥
        )
        if dist.get_rank() == 0:
            logger.info(f"Final Test mAP: {mAP:.3f}")
    else:
        top1, top5 = validate(
            test_loader, classes, device, model, video_head, config, n_class, logger,
            coapt_bias_enabled=coapt_bias_enabled,
            attribute_prompt_enabled=using_attribute_prompts,
            save_predictions_path=save_predictions_path,
            classnames=classnames,
            test_dataset=test_dataset,    # 🔥
        )
        if dist.get_rank() == 0:
            logger.info(f"Final Test Prec@1: {top1:.3f}")
            logger.info(f"Final Test Prec@5: {top5:.3f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
