"""
Zero-shot Video-Text Retrieval Evaluation
==========================================
利用已有的分类模型，在 UCF-101 / HMDB-51 上做「类别级视频-文本检索」实验。

两种文本 Query：
  - Base:  仅用类别名（如 "a video of pullup."）
  - Ours:  类别名 + 属性词（如 "a video of pullup. muscular strength upper body exercise."）

评估指标：
  - Text→Video Retrieval:  给定文本 query，检索视频，计算 R@1, R@5, R@10, MedianR
  - Video→Text Retrieval:  给定视频，检索文本 query，计算 R@1, R@5, R@10, MedianR

用法：
  python retrieval_eval.py --config your_config.yaml \
      --resume /path/to/best_model.pt \
      --log_time 001
"""

import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
from contextlib import suppress

import clip
from modules.video_clip import video_header
from modules.text_prompt import text_prompt
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, gather_labels
from utils.logger import setup_logger
from utils.Augmentation import get_augmentation
from Coviar.transforms import get_compress_augmentation

# ---- 复用 train.py 中的关键组件 ----
from train_comp_CoAPT import (
    CoAPTBiasAdapter,
    AllGather,
    allgather,
    load_attribute_descriptions,
    build_attribute_prompts,
    update_dict,
)


# =========================================================================
#  检索评估核心函数
# =========================================================================

def compute_similarity_matrix(video_features, text_features):
    """
    计算视频-文本相似度矩阵。
    
    Args:
        video_features: (N_videos, D) 归一化后的视频特征
        text_features:  (N_classes, D) 归一化后的文本特征
    
    Returns:
        sim_matrix: (N_videos, N_classes) 相似度矩阵
    """
    video_features = F.normalize(video_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    sim_matrix = video_features @ text_features.T  # (N_videos, N_classes)
    return sim_matrix


def retrieval_metrics(sim_matrix, video_labels, topk=(1, 5, 10)):
    """
    计算检索指标。
    
    类别级检索逻辑：
      - sim_matrix: (N_videos, N_classes), 每个视频与每个类别的相似度
      - video_labels: (N_videos,), 每个视频的真实类别标签
    
    Text→Video: 对于每个类别 c，把该类别的文本 query 作为查询，
                对所有视频按相似度排序，看属于类别 c 的视频排在第几。
    Video→Text: 对于每个视频，按相似度排序所有类别，看真实类别排在第几。
    
    Returns:
        t2v_metrics: dict, Text→Video 检索指标
        v2t_metrics: dict, Video→Text 检索指标
    """
    n_videos, n_classes = sim_matrix.shape
    
    # =====================
    # Video → Text Retrieval
    # =====================
    # 对于每个视频，在 n_classes 个文本中找到真实类别的排名
    v2t_ranks = []
    for i in range(n_videos):
        gt_class = video_labels[i].item()
        sim_row = sim_matrix[i]  # (n_classes,)
        # 降序排列，找到 gt_class 的排名（从 1 开始）
        sorted_indices = torch.argsort(sim_row, descending=True)
        rank = (sorted_indices == gt_class).nonzero(as_tuple=True)[0].item() + 1
        v2t_ranks.append(rank)
    
    v2t_ranks = torch.tensor(v2t_ranks, dtype=torch.float)
    
    v2t_metrics = {}
    for k in topk:
        v2t_metrics[f"R@{k}"] = (v2t_ranks <= k).float().mean().item() * 100.0
    v2t_metrics["MedianR"] = v2t_ranks.median().item()
    v2t_metrics["MeanR"] = v2t_ranks.mean().item()
    
    # =====================
    # Text → Video Retrieval
    # =====================
    # 对于每个类别 c，用它的文本 query 检索所有视频
    # 属于类别 c 的所有视频都是正样本，取最佳排名
    t2v_ranks = []
    for c in range(n_classes):
        sim_col = sim_matrix[:, c]  # (n_videos,)
        sorted_indices = torch.argsort(sim_col, descending=True)
        
        # 找到属于类别 c 的所有视频
        gt_mask = (video_labels == c)
        if gt_mask.sum() == 0:
            continue  # 该类别在验证集中没有视频，跳过
        
        # 取所有正样本中的最佳排名（最小排名）
        gt_positions = []
        for idx in range(len(sorted_indices)):
            if gt_mask[sorted_indices[idx]]:
                gt_positions.append(idx + 1)  # 排名从 1 开始
        
        best_rank = min(gt_positions)
        t2v_ranks.append(best_rank)
    
    t2v_ranks = torch.tensor(t2v_ranks, dtype=torch.float)
    
    t2v_metrics = {}
    for k in topk:
        t2v_metrics[f"R@{k}"] = (t2v_ranks <= k).float().mean().item() * 100.0
    t2v_metrics["MedianR"] = t2v_ranks.median().item()
    t2v_metrics["MeanR"] = t2v_ranks.mean().item()
    
    return t2v_metrics, v2t_metrics


def retrieval_metrics_instance_level(sim_matrix, video_labels, topk=(1, 5, 10)):
    """
    实例级 Text→Video 检索：
    对于每个类别 c，计算该类别文本检索到的视频中，
    各个正样本视频的平均排名，而非仅取最佳排名。
    这个指标更加严格。
    """
    n_videos, n_classes = sim_matrix.shape

    # Video → Text（与上面相同）
    v2t_ranks = []
    for i in range(n_videos):
        gt_class = video_labels[i].item()
        sim_row = sim_matrix[i]
        sorted_indices = torch.argsort(sim_row, descending=True)
        rank = (sorted_indices == gt_class).nonzero(as_tuple=True)[0].item() + 1
        v2t_ranks.append(rank)

    v2t_ranks = torch.tensor(v2t_ranks, dtype=torch.float)
    v2t_metrics = {}
    for k in topk:
        v2t_metrics[f"R@{k}"] = (v2t_ranks <= k).float().mean().item() * 100.0
    v2t_metrics["MedianR"] = v2t_ranks.median().item()
    v2t_metrics["MeanR"] = v2t_ranks.mean().item()

    # Text → Video（实例级：每个正样本的排名都算）
    t2v_all_ranks = []
    for c in range(n_classes):
        sim_col = sim_matrix[:, c]
        sorted_indices = torch.argsort(sim_col, descending=True)
        gt_mask = (video_labels == c)
        if gt_mask.sum() == 0:
            continue
        for idx in range(len(sorted_indices)):
            if gt_mask[sorted_indices[idx]]:
                t2v_all_ranks.append(idx + 1)

    t2v_all_ranks = torch.tensor(t2v_all_ranks, dtype=torch.float)
    t2v_metrics = {}
    for k in topk:
        t2v_metrics[f"R@{k}"] = (t2v_all_ranks <= k).float().mean().item() * 100.0
    t2v_metrics["MedianR"] = t2v_all_ranks.median().item()
    t2v_metrics["MeanR"] = t2v_all_ranks.mean().item()

    return t2v_metrics, v2t_metrics


# =========================================================================
#  构建 Base / Ours 文本特征
# =========================================================================

def build_base_text_features(classnames, clip_model, device):
    """
    Base Query: 仅使用类别名生成文本特征。
    模板: "a video of {classname}."
    """
    base_prompts = [f"a video of {name}." for name in classnames]
    text_tokens = torch.cat([clip.tokenize(p) for p in base_prompts]).to(device)
    
    with torch.no_grad():
        cls_features, token_features = clip_model.encode_text(text_tokens, return_token=True)
    
    return cls_features, token_features, base_prompts


def build_ours_text_features(classnames, attribute_cfg, clip_model, device, logger, dataset_name):
    """
    Ours Query: 使用类别名 + 属性词生成文本特征（Query Expansion）。
    """
    text_tokens, prompt_strings = build_attribute_prompts(
        classnames, attribute_cfg, logger, dataset_name
    )
    text_tokens = text_tokens.to(device)
    
    with torch.no_grad():
        cls_features, token_features = clip_model.encode_text(text_tokens, return_token=True)
    
    return cls_features, token_features, prompt_strings


# =========================================================================
#  提取所有视频特征
# =========================================================================

@torch.no_grad()
def extract_all_video_features(
    val_loader, clip_model, video_head, config, device,
    coapt_bias_enabled=False,
    coapt_module=None,
    attribute_fusion_enabled=False,
    text_features_for_fusion=None,
    base_weights=None,
    base_cls_feature=None,
    n_class=None,
):
    """
    遍历验证集，提取所有视频的特征向量。
    
    返回:
        all_video_features: (N, D) 所有视频的特征
        all_labels: (N,) 所有视频的标签
    """
    clip_model.eval()
    video_head.eval()
    
    all_features = []
    all_labels = []
    
    for i, (image, mv, residual, class_id) in enumerate(val_loader):
        image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
        mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
        residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])
        
        b, t, c_i, h, w = image.size()
        _, _, c_m, _, _ = mv.size()
        
        image_input = image.to(device).view(-1, c_i, h, w)
        mv_input = mv.to(device).view(-1, c_m, h, w)
        residual_input = residual.to(device).view(-1, c_i, h, w)
        
        # 编码视觉特征
        image_features, res_features, mvs_features = clip_model.encode_image(
            image_input, residual_input, mv_input
        )
        image_features = image_features.view(b, t, -1)
        res_features = res_features.view(b, t, -1)
        mvs_features = mvs_features.view(b, t, -1)
        
        if base_weights is None:
            base_weights = F.softmax(clip_model.beta, dim=0)
        
        # 融合多模态视觉特征为单一视频表征
        merged = (base_weights[0] * image_features +
                  base_weights[1] * res_features +
                  base_weights[2] * mvs_features)
        
        # 时序平均池化得到视频级表征 (B, D)
        video_feat = merged.mean(dim=1)
        
        # 如果启用 CoAPT bias，这里不做（因为 bias 依赖具体的文本 query）
        # CoAPT bias 会在后续计算相似度时动态应用
        
        all_features.append(video_feat.cpu())
        all_labels.append(class_id)
        
        if i % 50 == 0:
            print(f"  Extracting video features: [{i}/{len(val_loader)}]")
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


@torch.no_grad()
def extract_video_features_with_fusion_correct(
    val_loader, clip_model, video_head, config, device,
    text_token_features,   # (n_class, seq_len, D)
    text_cls_features,     # (n_class, D)
    base_weights, n_class,
):
    """
    正确的 Fusion 评估逻辑：
    - attribute_guided_fusion 用所有类别文本均值引导（类别无关，无标签泄露）
    - fused_feats (B, T, D) 直接时序平均得到视频表征 (B, D)
    - 最终与各类别 cls_feat 余弦相似度打分
    """
    clip_model.eval()
    video_head.eval()

    # 预计算：所有类别 token 特征的均值，shape: (1, seq_len, D)
    mean_token = text_token_features.mean(dim=0, keepdim=True).to(device)

    all_features = []
    all_labels   = []

    for i, (image, mv, residual, class_id) in enumerate(val_loader):
        # ---------- 数据预处理 ----------
        image    = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
        mv       = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
        residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])

        b, t, c_i, h, w = image.size()
        _, _, c_m, _, _ = mv.size()

        image_input    = image.to(device).view(-1, c_i, h, w)
        mv_input       = mv.to(device).view(-1, c_m, h, w)
        residual_input = residual.to(device).view(-1, c_i, h, w)

        # ---------- 编码视觉特征 ----------
        image_features, res_features, mvs_features = clip_model.encode_image(
            image_input, residual_input, mv_input
        )
        image_features = image_features.view(b, t, -1)   # (B, T, D)
        res_features   = res_features.view(b, t, -1)
        mvs_features   = mvs_features.view(b, t, -1)

        # ---------- 属性引导融合（均值文本引导，类别无关）----------
        mean_token_expanded = mean_token.expand(b, -1, -1)  # (B, seq_len, D)

        fused_feats, *_ = clip_model.attribute_guided_fusion(
            image_features, mvs_features, res_features,
            mean_token_expanded, base_weights,
        )
        # fused_feats: (B, T, D) — 已确认

        # ---------- 时序平均池化 → 视频级表征 ----------
        video_feat = fused_feats.mean(dim=1)   # (B, D)
        video_feat = F.normalize(video_feat, dim=-1)

        all_features.append(video_feat.cpu())
        all_labels.append(class_id)

        if i % 50 == 0:
            print(f"  Extracting fused features: [{i}/{len(val_loader)}]")

    all_features = torch.cat(all_features, dim=0)   # (N, D)
    all_labels   = torch.cat(all_labels, dim=0)     # (N,)

    # ---------- 与各类别 cls_feat 打分 ----------
    sim_matrix = compute_similarity_matrix(
        all_features.to(device), text_cls_features
    )
    return sim_matrix, all_labels





# =========================================================================
#  主函数
# =========================================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Zero-shot Video-Text Retrieval Evaluation")
    parser.add_argument('--config', '-cfg', type=str, required=True, help='config file path')
    parser.add_argument('--resume', type=str, required=True, help='path to trained checkpoint (best_model.pt)')
    parser.add_argument('--log_time', default='retrieval_eval')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--precision', choices=["amp", "fp16", "fp32"], default="fp32")
    parser.add_argument('--no-accumulation', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32, help='override batch size')
    parser.add_argument('--save_features', action='store_true', help='save extracted features to disk')
    parser.add_argument('--output_dir', type=str, default='retrieval_results', help='directory to save results')
    return parser.parse_args()


def main():
    args = get_parser()
    
    # ---- 初始化分布式（单卡也可以） ----
    init_distributed_mode(args)
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = DotMap(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Logger ----
    logger = setup_logger(output=str(output_dir), distributed_rank=dist.get_rank(), name='Retrieval')
    logger.info("=" * 60)
    logger.info("Zero-shot Video-Text Retrieval Evaluation")
    logger.info("=" * 60)
    
    # ---- 加载模型 ----
    residual_layers = config.network.get('residual_layers_to_use', None)
    mvs_layers = config.network.get('mvs_layers_to_use', None)
    
    model, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
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
    
    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()
    
    # ---- 配置 CoAPT Bias ----
    attribute_prompt_cfg = config.network.get('action_prompt', config.network.get('attribute_prompt', None))
    
    coapt_bias_cfg = config.network.get('coapt_bias', None)
    if coapt_bias_cfg is None and isinstance(attribute_prompt_cfg, dict):
        coapt_bias_cfg = attribute_prompt_cfg.get('coapt_bias', attribute_prompt_cfg.get('coapt', None))
    
    coapt_bias_enabled = False
    coapt_hidden_layers = None
    if isinstance(coapt_bias_cfg, dict):
        coapt_bias_enabled = bool(coapt_bias_cfg.get('enable', True))
        layers_cfg = coapt_bias_cfg.get('layers', None)
        if layers_cfg is not None:
            if isinstance(layers_cfg, (list, tuple)):
                coapt_hidden_layers = [int(v) for v in layers_cfg]
            elif isinstance(layers_cfg, int):
                coapt_hidden_layers = [int(layers_cfg)]
    else:
        coapt_bias_enabled = bool(coapt_bias_cfg) if coapt_bias_cfg is not None else False
    
    if coapt_bias_enabled:
        embed_dim = model.text_projection.shape[1]
        model.coapt_bias = CoAPTBiasAdapter(embed_dim, coapt_hidden_layers)
        logger.info("CoAPT bias meta-net enabled")
    else:
        model.coapt_bias = None
    
    # ---- 配置属性引导融合 ----
    attribute_fusion_cfg = config.network.get('attribute_guided_fusion', None)
    attribute_fusion_enabled = False
    fusion_kwargs = {}
    
    using_attribute_prompts = False
    if isinstance(attribute_prompt_cfg, dict) and attribute_prompt_cfg.get('enable', False):
        using_attribute_prompts = True
    
    if isinstance(attribute_fusion_cfg, dict):
        attribute_fusion_enabled = bool(attribute_fusion_cfg.get('enable', using_attribute_prompts))
        allowed_keys = {"hidden_dim", "num_heads", "dropout", "detach_text"}
        fusion_kwargs = {k: attribute_fusion_cfg[k] for k in allowed_keys if k in attribute_fusion_cfg}
    
    if attribute_fusion_enabled and not using_attribute_prompts:
        logger.warning("Attribute-guided fusion requires attribute prompts; disabling.")
        attribute_fusion_enabled = False
    
    if attribute_fusion_enabled:
        model.configure_attribute_guided_fusion(enable=True, **fusion_kwargs)
        logger.info("Attribute-guided fusion enabled")
    else:
        model.configure_attribute_guided_fusion(enable=False)
    
    # ---- 加载 Checkpoint ----
    logger.info(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(update_dict(checkpoint['model_state_dict']), strict=False)
    video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']), strict=False)
    logger.info("Checkpoint loaded successfully")
    
    model = model.to(device)
    video_head = video_head.to(device)
    model.eval()
    video_head.eval()
    
    # ---- 加载数据集 ----
    if config.data.modality in ['mv', 'residual', 'iframe']:
        transform_val = get_compress_augmentation(False, config)
    else:
        transform_val = get_augmentation(False, config)
    
    if config.data.modality in ['RGB', 'video']:
        if config.data.dataset == 'charades':
            from datasets.charades import Video_dataset
        else:
            from datasets.video import Video_dataset
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense,
        )
    elif config.data.modality in ['iframe', 'mv', 'residual']:
        from datasets.video3 import Video_dataset
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality, test_mode=True,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense,
            accumulate=(not args.no_accumulation),
        )
    
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size,
        num_workers=config.data.workers, shuffle=False, drop_last=False,
    )
    
    classnames = [c for _, c in val_data.classes]
    n_class = len(classnames)
    logger.info(f"Dataset: {config.data.dataset}, Classes: {n_class}, Videos: {len(val_data)}")
    
    # =========================================================================
    #  Step 1: 构建 Base 文本特征（仅类别名）
    # =========================================================================
    logger.info("-" * 60)
    logger.info("Building BASE text features (class name only)...")
    base_cls_feat, base_token_feat, base_prompts = build_base_text_features(
        classnames, model, device
    )
    logger.info(f"  Example: {base_prompts[0]}")
    
    # =========================================================================
    #  Step 2: 构建 Ours 文本特征（类别名 + 属性词）
    # =========================================================================
    ours_cls_feat, ours_token_feat, ours_prompts = None, None, None
    
    if using_attribute_prompts:
        logger.info("Building OURS text features (class name + attributes)...")
        ours_cls_feat, ours_token_feat, ours_prompts = build_ours_text_features(
            classnames, attribute_prompt_cfg, model, device, logger, config.data.dataset
        )
        logger.info(f"  Example: {ours_prompts[0]}")
    else:
        logger.warning("Attribute prompts not configured. Will only evaluate Base.")
    
    # =========================================================================
    #  Step 3: 提取视频特征
    # =========================================================================
    logger.info("-" * 60)
    logger.info("Extracting video features...")
    
    base_weights = F.softmax(model.beta, dim=0)
    logger.info(f"Fusion weights (iframe/res/mv): {base_weights.tolist()}")
    
    # --- 方式 A: 简单融合提取视频表征 ---
    video_features, video_labels = extract_all_video_features(
        val_loader, model, video_head, config, device,
        coapt_bias_enabled=coapt_bias_enabled,
        coapt_module=getattr(model, 'coapt_bias', None),
        base_weights=base_weights,
    )
    
    logger.info(f"Extracted {video_features.shape[0]} video features, dim={video_features.shape[1]}")
    
    # =========================================================================
    #  Step 4: 计算检索指标
    # =========================================================================
    results = {}
    
    # ---- BASE 检索 ----
    logger.info("=" * 60)
    logger.info("【BASE】Text-Video Retrieval (class name only)")
    logger.info("=" * 60)
    
    # 用 cls_feature 做相似度（与分类头一致）
    base_sim = compute_similarity_matrix(video_features.to(device), base_cls_feat)
    
    t2v_base, v2t_base = retrieval_metrics(base_sim, video_labels.to(device))
    t2v_base_inst, v2t_base_inst = retrieval_metrics_instance_level(base_sim, video_labels.to(device))
    
    logger.info("[BASE] Text→Video (best-rank per class):")
    for k, v in t2v_base.items():
        logger.info(f"  {k}: {v:.2f}")
    
    logger.info("[BASE] Video→Text:")
    for k, v in v2t_base.items():
        logger.info(f"  {k}: {v:.2f}")
    
    logger.info("[BASE] Text→Video (instance-level, stricter):")
    for k, v in t2v_base_inst.items():
        logger.info(f"  {k}: {v:.2f}")
    
    results['base'] = {
        'text2video': t2v_base,
        'video2text': v2t_base,
        'text2video_instance': t2v_base_inst,
        'video2text_instance': v2t_base_inst,
    }
    
    # ---- OURS 检索 ----
    if ours_cls_feat is not None:
        logger.info("=" * 60)
        logger.info("【OURS】Text-Video Retrieval (class name + attributes)")
        logger.info("=" * 60)
        
        ours_sim = compute_similarity_matrix(video_features.to(device), ours_cls_feat)
        
        t2v_ours, v2t_ours = retrieval_metrics(ours_sim, video_labels.to(device))
        t2v_ours_inst, v2t_ours_inst = retrieval_metrics_instance_level(ours_sim, video_labels.to(device))
        
        logger.info("[OURS] Text→Video (best-rank per class):")
        for k, v in t2v_ours.items():
            logger.info(f"  {k}: {v:.2f}")
        
        logger.info("[OURS] Video→Text:")
        for k, v in v2t_ours.items():
            logger.info(f"  {k}: {v:.2f}")
        
        logger.info("[OURS] Text→Video (instance-level, stricter):")
        for k, v in t2v_ours_inst.items():
            logger.info(f"  {k}: {v:.2f}")
        
        results['ours'] = {
            'text2video': t2v_ours,
            'video2text': v2t_ours,
            'text2video_instance': t2v_ours_inst,
            'video2text_instance': v2t_ours_inst,
        }
        
        # ---- 对比改善 ----
        logger.info("=" * 60)
        logger.info("【IMPROVEMENT】Ours vs Base")
        logger.info("=" * 60)
        
        logger.info("Text→Video Improvement:")
        for k in t2v_base.keys():
            if k in ['MedianR', 'MeanR']:
                delta = t2v_base[k] - t2v_ours[k]  # 排名越低越好
                symbol = "↓" if delta > 0 else "↑"
                logger.info(f"  {k}: {t2v_base[k]:.2f} → {t2v_ours[k]:.2f} ({symbol}{abs(delta):.2f})")
            else:
                delta = t2v_ours[k] - t2v_base[k]  # Recall 越高越好
                symbol = "↑" if delta > 0 else "↓"
                logger.info(f"  {k}: {t2v_base[k]:.2f} → {t2v_ours[k]:.2f} ({symbol}{abs(delta):.2f})")
        
        logger.info("Video→Text Improvement:")
        for k in v2t_base.keys():
            if k in ['MedianR', 'MeanR']:
                delta = v2t_base[k] - v2t_ours[k]
                symbol = "↓" if delta > 0 else "↑"
                logger.info(f"  {k}: {v2t_base[k]:.2f} → {v2t_ours[k]:.2f} ({symbol}{abs(delta):.2f})")
            else:
                delta = v2t_ours[k] - v2t_base[k]
                symbol = "↑" if delta > 0 else "↓"
                logger.info(f"  {k}: {v2t_base[k]:.2f} → {v2t_ours[k]:.2f} ({symbol}{abs(delta):.2f})")
    
    # =========================================================================
    #  Step 5 (可选): 使用属性引导融合的检索
    # =========================================================================
    if attribute_fusion_enabled and ours_token_feat is not None:
        logger.info("=" * 60)
        logger.info("【OURS + Fusion】Attribute-Guided Fusion Retrieval")
        logger.info("=" * 60)
        
        fusion_sim, fusion_labels = extract_video_features_with_fusion_correct(
            val_loader, model, video_head, config, device,
            ours_token_feat, ours_cls_feat, base_weights, n_class,
        )
        
        t2v_fusion, v2t_fusion = retrieval_metrics(fusion_sim, fusion_labels)
        t2v_fusion_inst, v2t_fusion_inst = retrieval_metrics_instance_level(fusion_sim, fusion_labels)
        
        logger.info("[OURS+Fusion] Text→Video:")
        for k, v in t2v_fusion.items():
            logger.info(f"  {k}: {v:.2f}")
        
        logger.info("[OURS+Fusion] Video→Text:")
        for k, v in v2t_fusion.items():
            logger.info(f"  {k}: {v:.2f}")
        
        results['ours_fusion'] = {
            'text2video': t2v_fusion,
            'video2text': v2t_fusion,
            'text2video_instance': t2v_fusion_inst,
            'video2text_instance': v2t_fusion_inst,
        }
        
        # 与 Base 对比
        logger.info("Fusion vs Base Improvement (Text→Video):")
        for k in t2v_base.keys():
            if k in ['MedianR', 'MeanR']:
                delta = t2v_base[k] - t2v_fusion[k]
                symbol = "↓" if delta > 0 else "↑"
                logger.info(f"  {k}: {t2v_base[k]:.2f} → {t2v_fusion[k]:.2f} ({symbol}{abs(delta):.2f})")
            else:
                delta = t2v_fusion[k] - t2v_base[k]
                symbol = "↑" if delta > 0 else "↓"
                logger.info(f"  {k}: {t2v_base[k]:.2f} → {t2v_fusion[k]:.2f} ({symbol}{abs(delta):.2f})")
    
    # =========================================================================
    #  保存结果
    # =========================================================================
    result_file = output_dir / "retrieval_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {result_file}")
    
    # 保存 Prompt 对比
    prompt_file = output_dir / "prompt_comparison.txt"
    with open(prompt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Base vs Ours Prompt Comparison\n")
        f.write("=" * 80 + "\n\n")
        for i, name in enumerate(classnames):
            f.write(f"Class [{i}]: {name}\n")
            f.write(f"  Base: {base_prompts[i]}\n")
            if ours_prompts:
                f.write(f"  Ours: {ours_prompts[i]}\n")
            f.write("\n")
    logger.info(f"Prompt comparison saved to {prompt_file}")
    
    # 保存特征（可选）
    if args.save_features:
        feat_file = output_dir / "features.pt"
        save_dict = {
            'video_features': video_features,
            'video_labels': video_labels,
            'base_cls_feat': base_cls_feat.cpu(),
            'classnames': classnames,
        }
        if ours_cls_feat is not None:
            save_dict['ours_cls_feat'] = ours_cls_feat.cpu()
        torch.save(save_dict, feat_file)
        logger.info(f"Features saved to {feat_file}")
    
    # =========================================================================
    #  打印最终汇总表格
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    header = f"{'Method':<20} {'T2V R@1':>8} {'T2V R@5':>8} {'T2V R@10':>9} {'T2V MdR':>8} | {'V2T R@1':>8} {'V2T R@5':>8} {'V2T R@10':>9} {'V2T MdR':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    
    for method_name, method_results in results.items():
        t2v = method_results['text2video']
        v2t = method_results['video2text']
        row = (
            f"{method_name:<20} "
            f"{t2v.get('R@1', 0):>8.2f} "
            f"{t2v.get('R@5', 0):>8.2f} "
            f"{t2v.get('R@10', 0):>9.2f} "
            f"{t2v.get('MedianR', 0):>8.1f} | "
            f"{v2t.get('R@1', 0):>8.2f} "
            f"{v2t.get('R@5', 0):>8.2f} "
            f"{v2t.get('R@10', 0):>9.2f} "
            f"{v2t.get('MedianR', 0):>8.1f}"
        )
        logger.info(row)
    
    logger.info("=" * 80)
    logger.info("Done!")


if __name__ == '__main__':
    main()