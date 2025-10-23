#!/usr/bin/env python3
"""Attribute search for BIKE using DDP multi-GPU training."""
from datetime import timedelta
import argparse
import json
import os
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torchvision.transforms as T
import yaml
from dotmap import DotMap

from Coviar.transforms import GroupCenterCrop, GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupScale

import clip
from datasets.video3 import Video_dataset
from modules.video_clip import video_header
from utils.utils import AverageMeter, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-based attribute search for BIKE")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--attributes", required=True, help="属性JSON文件")
    parser.add_argument("--checkpoint", default="", help="可选模型检查点")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的批大小")
    parser.add_argument("--num-workers", type=int, default=None, help="数据加载线程数")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--log-interval", type=int, default=20, help="日志打印间隔")
    parser.add_argument("--max-motion", type=int, default=1, help="动属性最大选择数")
    parser.add_argument("--max-appearance", type=int, default=1, help="外观属性最大选择数")
    parser.add_argument("--n-ctx", type=int, default=8, help="类别上下文token数量")
    parser.add_argument("--n-att", type=int, default=4, help="属性上下文token数量")
    parser.add_argument("--prompt-lr", type=float, default=1e-3, help="提示参数学习率")
    parser.add_argument("--weight-lr", type=float, default=1e-3, help="属性权重学习率")
    parser.add_argument("--weight-entropy", type=float, default=5e-2, help="属性权重熵正则系数")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="提示参数权重衰减")
    parser.add_argument("--topk", type=int, default=5, help="最终输出Top-K组合")
    parser.add_argument("--combo-chunk-size", type=int, default=4, help="一次并行编码的属性组合数量")
    parser.add_argument("--output", default="", help="可选结果保存路径")
    
    # DDP相关参数
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    parser.add_argument("--world-size", type=int, default=1, help="分布式训练进程数")
    parser.add_argument("--dist-backend", default="nccl", help="分布式后端")
    
    return parser.parse_args()


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    
    # ✅ 修复：增加超时时间
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=2)  # 从默认10分钟增加到2小时
    )
    dist.barrier()
    
    return True, rank, world_size, local_rank

def evaluate_all_combos_batch(
    loader: DataLoader,
    model: nn.Module,
    video_head: nn.Module,
    config: DotMap,
    combo_cls_embs: torch.Tensor,
    attr_features: torch.Tensor,
    base_weights: torch.Tensor,
    rank: int,
) -> List[float]:
    """✅ 新增：批量评估所有组合，减少通信次数"""
    K = combo_cls_embs.size(0)
    local_top1_list = []
    
    print(f"[Rank {rank}] 开始评估 {K} 个属性组合...")
    
    with torch.no_grad():
        for idx in range(K):
            top1 = AverageMeter()
            one_hot = torch.zeros_like(base_weights)
            one_hot[idx] = 1.0
            
            for batch in loader:
                logits, labels = compute_video_logits(
                    model, video_head, batch, config, combo_cls_embs, attr_features, one_hot
                )
                prec1 = accuracy(logits, labels, topk=(1,))[0]
                top1.update(prec1.item(), labels.size(0))
            
            local_top1_list.append(top1.avg)
            
            if (idx + 1) % 10 == 0:
                print(f"[Rank {rank}] 已评估 {idx+1}/{K} 个组合")
    
    # 一次性同步所有结果
    if dist.is_initialized():
        print(f"[Rank {rank}] 开始同步评估结果...")
        acc_tensor = torch.tensor(local_top1_list, device=f'cuda:{rank}')
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
        print(f"[Rank {rank}] 同步完成")
        return acc_tensor.cpu().tolist()
    
    return local_top1_list


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0


def load_config(path: str) -> DotMap:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DotMap(cfg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transforms(input_size: int) -> Dict[str, T.Compose]:
    iframe_scales = [1.0, 0.875, 0.75, 0.66]
    mv_scales = [1.0, 0.875, 0.75]
    residual_scales = [1.0, 0.875, 0.75]

    return {
        "iframe": T.Compose([GroupMultiScaleCrop(input_size, iframe_scales), GroupRandomHorizontalFlip(is_mv=False)]),
        "mv": T.Compose([GroupMultiScaleCrop(input_size, mv_scales), GroupRandomHorizontalFlip(is_mv=True)]),
        "residual": T.Compose([GroupMultiScaleCrop(input_size, residual_scales), GroupRandomHorizontalFlip(is_mv=False)]),
    }


def build_val_transform(input_size: int) -> T.Compose:
    scale_size = int(round(input_size / 224 * 256))
    return T.Compose([GroupScale(scale_size), GroupCenterCrop(input_size)])


def combinations_for_category(attrs: Sequence[str], max_sel: int) -> List[Tuple[str, ...]]:
    if max_sel <= 0:
        return [tuple()]
    upper = min(max_sel, len(attrs))
    combos: List[Tuple[str, ...]] = []
    for r in range(1, upper + 1):
        combos.extend(combinations(attrs, r))
    return combos if combos else [tuple()]


def build_attribute_combinations(
    motion_attrs: Sequence[str],
    appearance_attrs: Sequence[str],
    max_motion: int,
    max_appearance: int,
) -> Tuple[List[Tuple[str, ...]], List[Dict[str, List[str]]]]:
    motion_sets = combinations_for_category(motion_attrs, max_motion)
    appearance_sets = combinations_for_category(appearance_attrs, max_appearance)

    combos: List[Tuple[str, ...]] = []
    meta: List[Dict[str, List[str]]] = []
    for m in motion_sets:
        for a in appearance_sets:
            if not m and not a:
                continue
            combo = tuple(list(m) + list(a))
            combos.append(combo)
            meta.append({"motion": list(m), "appearance": list(a)})
    return combos, meta


def format_attribute_text(motion: Sequence[str], appearance: Sequence[str]) -> str:
    """Construct attribute string with shared context placeholders."""
    parts: List[str] = ["[Ta1]", "[Ta2]"]
    motion_part = " ".join(motion).strip()
    if motion_part:
        parts.append(motion_part)
    parts.extend(["[Tb1]", "[Tb2]"])
    appearance_part = " ".join(appearance).strip()
    if appearance_part:
        parts.append(appearance_part)
    return " ".join(parts).strip()


class TextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module, n_cls: int):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_cls = n_cls

    def forward(self, prompts_list: List[torch.Tensor], tokenized_prompts_list: List[torch.LongTensor]) -> torch.Tensor:
        encoded: List[torch.Tensor] = []
        for prompts, tokenized in zip(prompts_list, tokenized_prompts_list):
            tokenized = tokenized.to(prompts.device)
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x).type(self.dtype)
            pooled = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.text_projection
            encoded.append(pooled.unsqueeze(0))
        return torch.cat(encoded, dim=0)

def convert_to_gerund(classname: str) -> str:
    """将HMDB51类别名转换为自然语言"""
    # 特殊映射表
    special_cases = {
        "brush_hair": "brushing hair",
        "cartwheel": "doing a cartwheel",
        "catch": "catching",
        "chew": "chewing",
        "clap": "clapping",
        "climb": "climbing",
        "climb_stairs": "climbing stairs",
        "dive": "diving",
        "draw_sword": "drawing a sword",
        "dribble": "dribbling",
        "drink": "drinking",
        "eat": "eating",
        "fall_floor": "falling on the floor",
        "fencing": "fencing",
        "flic_flac": "doing a flic flac",
        "golf": "playing golf",
        "handstand": "doing a handstand",
        "hit": "hitting",
        "hug": "hugging",
        "jump": "jumping",
        "kick": "kicking",
        "kick_ball": "kicking a ball",
        "kiss": "kissing",
        "laugh": "laughing",
        "pick": "picking",
        "pour": "pouring",
        "pullup": "doing pullups",
        "punch": "punching",
        "push": "pushing",
        "pushup": "doing pushups",
        "ride_bike": "riding a bike",
        "ride_horse": "riding a horse",
        "run": "running",
        "shake_hands": "shaking hands",
        "shoot_ball": "shooting a ball",
        "shoot_bow": "shooting a bow",
        "shoot_gun": "shooting a gun",
        "sit": "sitting",
        "situp": "doing situps",
        "smile": "smiling",
        "smoke": "smoking",
        "somersault": "doing a somersault",
        "stand": "standing",
        "swing_baseball": "swinging a baseball bat",
        "sword": "sword fighting",
        "sword_exercise": "sword exercise",
        "talk": "talking",
        "throw": "throwing",
        "turn": "turning",
        "walk": "walking",
        "wave": "waving",
    }
    
    return special_cases.get(classname, classname.replace("_", " "))


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        classnames: Sequence[str],
        combinations: Sequence[Tuple[str, ...]],
        n_ctx: int,
        n_att: int,
        class_token_position: str = "end",
    ) -> None:
        super().__init__()
        if n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if n_att <= 0:
            raise ValueError("n_att must be positive")
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.n_att = n_att
        self.dtype = clip_model.dtype
        self.class_token_position = class_token_position
        ctx_dim = clip_model.ln_final.weight.shape[0]

        prompt_prefix = " ".join(["X"] * n_ctx)
        attribute_prefix = " ".join(["X"] * n_att)
        classnames = [convert_to_gerund(name) for name in classnames]

        self.ctx_list = nn.ParameterList()
        self.att_ctx_list = nn.ModuleList()
        self.tokenized_prompt_list: List[torch.LongTensor] = []
        self.embedding_lists: List[torch.Tensor] = []
        self.combinations = list(combinations)

        for combo in self.combinations:
            ctx_vectors = nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=self.dtype))
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx_list.append(ctx_vectors)

            att_params = nn.ParameterList()
            combo_chunks: List[str] = []
            for attr in combo:
                att_vec = nn.Parameter(torch.empty(n_att, ctx_dim, dtype=self.dtype))
                nn.init.normal_(att_vec, std=0.01)
                att_params.append(att_vec)
                combo_chunks.append(f"{attribute_prefix} {attr}")
            self.att_ctx_list.append(att_params)

            attribute_text = " ".join(combo_chunks).strip()
            if attribute_text:
                base_prompt = f"{attribute_text} {prompt_prefix}"
            else:
                base_prompt = prompt_prefix
            base_prompt = base_prompt.strip()
            if base_prompt:
                prompts = [f"{base_prompt} This is a video about {name}." for name in classnames]
            else:
                prompts = [f"This is a video about {name}." for name in classnames]

            tokenized_prompt = torch.cat([clip.tokenize(p) for p in prompts])
            self.tokenized_prompt_list.append(tokenized_prompt)
            with torch.no_grad():
                tokenized_prompt_device = tokenized_prompt.to(clip_model.token_embedding.weight.device)
                embedding = clip_model.token_embedding(tokenized_prompt_device).to(dtype=torch.float16).cpu()
            self.embedding_lists.append(embedding)

        init_noise = torch.randn(len(self.combinations), dtype=torch.float32) * 1e-2
        self.att_weight = nn.Parameter(init_noise)

    def construct_prompt(self, idx: int, device: torch.device) -> torch.Tensor:
        embedding = self.embedding_lists[idx].to(device=device, dtype=self.dtype)
        ctx = self.ctx_list[idx].to(device)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        att_ctxs = [param.to(device) for param in self.att_ctx_list[idx]]
        if self.class_token_position != "end":
            raise ValueError("Only class_token_position='end' is supported")

        prefix = embedding[:, :1, :]
        pieces = [prefix]
        cursor = 1
        for att_ctx in att_ctxs:
            if att_ctx.dim() == 2:
                att_ctx = att_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            pieces.append(att_ctx)
            word_slice = embedding[:, cursor + self.n_att : cursor + self.n_att + 1, :]
            pieces.append(word_slice)
            cursor += self.n_att + 1

        ctx_expanded = ctx
        pieces.append(ctx_expanded)
        suffix = embedding[:, cursor + self.n_ctx :, :]
        pieces.append(suffix)
        prompt = torch.cat(pieces, dim=1)
        return prompt

    def forward(self) -> List[torch.Tensor]:
        device = self.att_weight.device
        return [self.construct_prompt(idx, device) for idx in range(len(self.combinations))]


class AttributePromptModel(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        classnames: Sequence[str],
        combinations: Sequence[Tuple[str, ...]],
        combo_meta: Sequence[Dict[str, List[str]]],
        n_ctx: int,
        n_att: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.prompt_learner = PromptLearner(clip_model, classnames, combinations, n_ctx, n_att)
        self.text_encoder = TextEncoder(clip_model, len(classnames))
        self.chunk_size = max(1, chunk_size)
        self.combo_meta = list(combo_meta)

        # Pre-tokenize attribute prompts for each combination to reuse during training.
        self.attr_texts: List[str] = []
        self.attr_tokenized_list: List[torch.LongTensor] = []
        self.attr_embedding_list: List[torch.Tensor] = []
        for meta in self.combo_meta:
            attr_text = format_attribute_text(meta["motion"], meta["appearance"])
            if not attr_text:
                attr_text = "[Ta1] [Ta2] [Tb1] [Tb2]"
            tokenized = clip.tokenize(attr_text)
            with torch.no_grad():
                tokenized_device = tokenized.to(clip_model.token_embedding.weight.device)
                embedding = clip_model.token_embedding(tokenized_device).to(dtype=torch.float16).cpu()
            self.attr_texts.append(attr_text)
            self.attr_tokenized_list.append(tokenized)
            self.attr_embedding_list.append(embedding)

    def get_text_features(
        self,
        detach_prompt: bool,
        freeze_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.prompt_learner.att_weight.device
        total = len(self.prompt_learner.combinations)
        chunk = min(self.chunk_size, total)
        feature_chunks: List[torch.Tensor] = []

        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            prompts_batch: List[torch.Tensor] = []
            tokenized_batch: List[torch.LongTensor] = []
            for idx in range(start, end):
                prompts_batch.append(self.prompt_learner.construct_prompt(idx, device))
                tokenized_batch.append(self.prompt_learner.tokenized_prompt_list[idx].to(device))
            batch_feat = self.text_encoder(prompts_batch, tokenized_batch)
            feature_chunks.append(batch_feat)

        text_features = torch.cat(feature_chunks, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if detach_prompt:
            text_features = text_features.detach()

        attr_prompts: List[torch.Tensor] = []
        attr_tokenized_batch: List[torch.LongTensor] = []
        for idx in range(total):
            attr_prompts.append(self.attr_embedding_list[idx].to(device=device, dtype=self.prompt_learner.dtype))
            attr_tokenized_batch.append(self.attr_tokenized_list[idx].to(device))
        attr_features = self.text_encoder(attr_prompts, attr_tokenized_batch)
        attr_features = attr_features / attr_features.norm(dim=-1, keepdim=True)
        if detach_prompt:
            attr_features = attr_features.detach()

        weights = F.softmax(self.prompt_learner.att_weight, dim=0)
        if freeze_weights:
            weights = weights.detach()

        weighted = torch.einsum("k,knd->nd", weights, text_features)
        weighted = weighted / weighted.norm(dim=-1, keepdim=True)
        return weighted, text_features, weights, attr_features


def compute_video_logits(
    model: nn.Module,
    video_head: nn.Module,
    batch,
    config: DotMap,
    combo_cls_embs: torch.Tensor,
    attr_features: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = weights.device
    combo_cls_embs = combo_cls_embs.to(device)
    attr_features = attr_features.to(device)
    weights = weights.to(device)

    images, residuals, mvs, labels = batch
    images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])
    mvs = mvs.view((-1, config.data.num_segments, 2) + mvs.size()[-2:])
    residuals = residuals.view((-1, config.data.num_segments, 3) + residuals.size()[-2:])

    b, t, c_i, h, w = images.size()
    _, _, c_m, _, _ = mvs.size()

    images = images.to(device).view(-1, c_i, h, w)
    mvs = mvs.to(device).view(-1, c_m, h, w)
    residuals = residuals.to(device).view(-1, c_i, h, w)
    labels = labels.to(device)

    with torch.no_grad():
        image_feats, res_feats, mv_feats = model.encode_image(images, residuals, mvs)
        image_feats = image_feats.view(b, t, -1)
        res_feats = res_feats.view(b, t, -1)
        mv_feats = mv_feats.view(b, t, -1)

        if hasattr(video_head, "agg_video_feat") and video_head is not None:
            image_feats = video_head.agg_video_feat(image_feats)
            res_feats = video_head.agg_video_feat(res_feats)
            mv_feats = video_head.agg_video_feat(mv_feats)

    image_feat = image_feats.mean(dim=1)
    res_feat = res_feats.mean(dim=1)
    mv_feat = mv_feats.mean(dim=1)

    eps = 1e-6
    image_unit = image_feat / (image_feat.norm(dim=-1, keepdim=True) + eps)
    res_unit = res_feat / (res_feat.norm(dim=-1, keepdim=True) + eps)
    mv_unit = mv_feat / (mv_feat.norm(dim=-1, keepdim=True) + eps)

    combo_logits: List[torch.Tensor] = []
    for idx in range(combo_cls_embs.size(0)):
        attr_vec = attr_features[idx].to(device)
        attr_unit = attr_vec / (attr_vec.norm(dim=-1, keepdim=True) + eps)

        dot_mv = (mv_unit * attr_unit).sum(dim=-1)
        dot_res = (res_unit * attr_unit).sum(dim=-1)
        motion_logits = torch.stack([dot_mv, dot_res], dim=-1)
        attention_motion = torch.softmax(motion_logits, dim=-1)

        dot_img = (image_unit * attr_unit).sum(dim=-1)
        appearance_logits = torch.stack([dot_img, torch.zeros_like(dot_img)], dim=-1)
        attention_appearance = torch.softmax(appearance_logits, dim=-1)[..., 0].unsqueeze(-1)

        # Fuse per-modality cues using learned attribute attention.
        fused_visual = (
            attention_motion[..., 0:1] * mv_unit
            + attention_motion[..., 1:2] * res_unit
            + attention_appearance * image_unit
        )
        fused_visual = fused_visual / (fused_visual.norm(dim=-1, keepdim=True) + eps)

        cls_emb = combo_cls_embs[idx].to(device)
        cls_emb = cls_emb / (cls_emb.norm(dim=-1, keepdim=True) + eps)
        logits_i = fused_visual @ cls_emb.t()
        combo_logits.append(logits_i)

    stacked_logits = torch.stack(combo_logits, dim=0)  # (K, B, num_classes)
    final_logits = torch.einsum("k,kbn->bn", weights, stacked_logits)
    return final_logits, labels


def evaluate_accuracy(
    loader: DataLoader,
    model: nn.Module,
    video_head: nn.Module,
    config: DotMap,
    combo_cls_embs: torch.Tensor,
    attr_features: torch.Tensor,
    weights: torch.Tensor,
    rank: int,
) -> float:
    top1 = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            logits, labels = compute_video_logits(
                model, video_head, batch, config, combo_cls_embs, attr_features, weights
            )
            prec1 = accuracy(logits, labels, topk=(1,))[0]
            top1.update(prec1.item(), labels.size(0))
    
    # 同步所有进程的结果
    if dist.is_initialized():
        acc_tensor = torch.tensor([top1.avg], device=f'cuda:{rank}')
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
        return acc_tensor.item()
    return top1.avg


def optimize_weights_epoch(
    prompt_model: nn.Module,
    optimizer_weight: torch.optim.Optimizer,
    val_loader: DataLoader,
    model: nn.Module,
    video_head: nn.Module,
    config: DotMap,
    criterion: nn.Module,
    rank: int,
    weight_entropy: float,
    is_distributed: bool,
) -> float:
    prompt_model.eval()
    if is_distributed:
        prompt_learner = prompt_model.module.prompt_learner
        _fused, combo_cls_embs, _weights_cached, attr_features = prompt_model.module.get_text_features(
            detach_prompt=True, freeze_weights=False
        )
    else:
        prompt_learner = prompt_model.prompt_learner
        _fused, combo_cls_embs, _weights_cached, attr_features = prompt_model.get_text_features(
            detach_prompt=True, freeze_weights=False
        )

    loss_meter = AverageMeter()
    loss_sum = 0.0
    sample_count = 0
    optimizer_weight.zero_grad()
    num_batches = max(1, len(val_loader))

    combo_cls_embs = combo_cls_embs.detach()
    attr_features = attr_features.detach()

    for batch in val_loader:
        weights = F.softmax(prompt_learner.att_weight, dim=0)
        logits, labels = compute_video_logits(
            model, video_head, batch, config, combo_cls_embs, attr_features, weights
        )
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), labels.size(0))
        loss_sum += loss.item() * labels.size(0)
        sample_count += labels.size(0)
    loss.backward()

    if weight_entropy > 0.0:
        weights_entropy = F.softmax(prompt_learner.att_weight, dim=0)
        entropy = -(weights_entropy * weights_entropy.clamp(min=1e-6).log()).sum()
        (-weight_entropy * entropy).backward()

    optimizer_weight.step()

    avg_loss = loss_meter.avg
    if dist.is_initialized():
        device = weights.device
        total_loss_tensor = torch.tensor([loss_sum], device=device)
        total_count_tensor = torch.tensor([sample_count], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / total_count_tensor.clamp(min=1)).item()
        dist.barrier()

    return avg_loss


def train_attribute_search(args: argparse.Namespace) -> None:
    # 初始化分布式环境
    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if is_distributed else 'cuda')
    
    if is_main_process(rank):
        print(f"分布式训练: {is_distributed}, Rank: {rank}, World Size: {world_size}")
    
    config = load_config(args.config)
    if config.network.interaction != "DP":
        raise NotImplementedError("当前脚本仅支持 interaction=DP")

    batch_size = args.batch_size or config.data.batch_size
    num_workers = args.num_workers if args.num_workers is not None else config.data.workers

    set_seed(config.seed + rank)  # 每个进程使用不同的随机种子
    torch.backends.cudnn.benchmark = True

    # 加载模型
    model, clip_state = clip.load(
        config.network.arch,
        device="cpu",
        jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st=config.network.joint_st,
        residual_layers_to_use=config.network.get("residual_layers_to_use", None),
        mvs_layers_to_use=config.network.get("mvs_layers_to_use", None),
    )
    video_head = video_header(config.network.sim_header, config.network.interaction, clip_state)

    model = model.to(device).float()
    video_head = video_head.to(device)

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"未找到检查点: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        video_head.load_state_dict(ckpt["fusion_model_state_dict"], strict=False)

    for param in model.parameters():
        param.requires_grad_(False)
    for param in video_head.parameters():
        param.requires_grad_(False)

    input_size = config.data.get("input_size", 224)
    dense_sample = config.data.get("dense", False)

    # 创建数据集
    train_dataset = Video_dataset(
        config.data.train_root,
        config.data.train_list,
        config.data.label_list,
        num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        random_shift=config.data.random_shift,
        transform=build_train_transforms(input_size),
        dense_sample=dense_sample,
    )
    val_dataset = Video_dataset(
        config.data.val_root,
        config.data.val_list,
        config.data.label_list,
        random_shift=False,
        num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=build_val_transform(input_size),
        dense_sample=dense_sample,
    )

    # 使用分布式采样器
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 加载属性配置
    with open(args.attributes, "r", encoding="utf-8") as f:
        attr_json = json.load(f)
    motion_attrs = attr_json["attributes"].get("motion", [])
    appearance_attrs = attr_json["attributes"].get("appearance", [])

    combos, combo_meta = build_attribute_combinations(
        motion_attrs, appearance_attrs, args.max_motion, args.max_appearance
    )
    if not combos:
        raise ValueError("未能构建有效的属性组合")

    if is_main_process(rank):
        print(f"总共生成 {len(combos)} 个属性组合")

    classnames = [name for _, name in train_dataset.classes]
    prompt_model = AttributePromptModel(
        model, classnames, combos, combo_meta, args.n_ctx, args.n_att, args.combo_chunk_size
    ).to(device)

    # 使用DDP包装prompt_model
    if is_distributed:
        prompt_model = DDP(
            prompt_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # 获取实际的prompt_learner（处理DDP包装）
    prompt_learner = prompt_model.module.prompt_learner if is_distributed else prompt_model.prompt_learner

    prompt_params = [param for name, param in prompt_learner.named_parameters() if name != "att_weight"]
    if not prompt_params:
        raise ValueError("提示参数集合为空")

    optimizer_prompt = torch.optim.Adam(prompt_params, lr=args.prompt_lr, weight_decay=args.weight_decay)
    optimizer_weight = torch.optim.Adam([prompt_learner.att_weight], lr=args.weight_lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
        
        prompt_model.train()
        train_loss_meter = AverageMeter()

        for step, batch in enumerate(train_loader):
            # 更新提示向量
            if is_distributed:
                _fused_emb, combo_cls_embs, weights, attr_features = prompt_model.module.get_text_features(
                    detach_prompt=False, freeze_weights=True
                )
            else:
                _fused_emb, combo_cls_embs, weights, attr_features = prompt_model.get_text_features(
                    detach_prompt=False, freeze_weights=True
                )
            
            logits, labels = compute_video_logits(
                model, video_head, batch, config, combo_cls_embs, attr_features, weights
            )
            loss = criterion(logits, labels)
            optimizer_prompt.zero_grad()
            loss.backward()
            optimizer_prompt.step()
            train_loss_meter.update(loss.item(), labels.size(0))

            if (step + 1) % args.log_interval == 0 and is_main_process(rank):
                print(
                    f"Epoch {epoch+1}/{args.epochs} Step {step+1}/{len(train_loader)} "
                    f"train_loss={train_loss_meter.avg:.4f}"
                )

        # 基于完整验证集优化权重
        weight_loss_val = optimize_weights_epoch(
            prompt_model,
            optimizer_weight,
            val_loader,
            model,
            video_head,
            config,
            criterion,
            rank,
            args.weight_entropy,
            is_distributed,
        )

        # 评估
        prompt_model.eval()
        if is_distributed:
            _fused_emb_eval, combo_cls_embs_eval, weights_eval, attr_features_eval = prompt_model.module.get_text_features(
                detach_prompt=True, freeze_weights=False
            )
        else:
            _fused_emb_eval, combo_cls_embs_eval, weights_eval, attr_features_eval = prompt_model.get_text_features(
                detach_prompt=True, freeze_weights=False
            )
        
        val_acc = evaluate_accuracy(
            val_loader,
            model,
            video_head,
            config,
            combo_cls_embs_eval,
            attr_features_eval,
            weights_eval,
            rank,
        )
        
        if is_main_process(rank):
            weight_list = weights_eval.detach().cpu().tolist()
            print(
                f"[Epoch {epoch+1}] val@1={val_acc:.2f} weight_loss={weight_loss_val:.4f} "
                f"best_weight={max(weight_list):.4f} avg_weight={sum(weight_list)/len(weight_list):.4f}"
            )

    # ✅ 修复：最终评估使用批量方法
    if is_main_process(rank):
        print("\n" + "="*50)
        print("开始最终评估...")
        print("="*50)
    
    # 确保所有进程同步
    if dist.is_initialized():
        dist.barrier()
    
    prompt_model.eval()
    if is_distributed:
        _fused_emb_final, combo_cls_embs_final, weights_final, attr_features_final = (
            prompt_model.module.get_text_features(
                detach_prompt=True, freeze_weights=False
            )
        )
    else:
        _fused_emb_final, combo_cls_embs_final, weights_final, attr_features_final = (
            prompt_model.get_text_features(
                detach_prompt=True, freeze_weights=False
            )
        )
    
    # 评估融合结果
    fused_acc = evaluate_accuracy(
        val_loader,
        model,
        video_head,
        config,
        combo_cls_embs_final,
        attr_features_final,
        weights_final,
        rank,
    )

    # ✅ 使用批量评估方法
    combo_accs = evaluate_all_combos_batch(
        val_loader,
        model,
        video_head,
        config,
        combo_cls_embs_final,
        attr_features_final,
        weights_final,
        rank,
    )
    
    # 仅主进程整理和保存结果
    if is_main_process(rank):
        combo_results: List[Dict[str, object]] = []
        for idx, (meta, acc) in enumerate(zip(combo_meta, combo_accs)):
            combo_results.append({
                "motion": meta["motion"],
                "appearance": meta["appearance"],
                "weight": weights_final[idx].detach().item(),
                "val_top1": acc,
            })
        combo_results.sort(key=lambda x: x["val_top1"], reverse=True)

        top_results = combo_results[: min(args.topk, len(combo_results))]
        print("\n最优属性组合：")
        for rank_idx, item in enumerate(top_results, start=1):
            motion_str = "/".join(item["motion"]) if item["motion"] else "-"
            appearance_str = "/".join(item["appearance"]) if item["appearance"] else "-"
            print(
                f"Top{rank_idx}: motion=[{motion_str}] appearance=[{appearance_str}] "
                f"weight={item['weight']:.4f} val@1={item['val_top1']:.2f}"
            )
        print(f"加权组合 val@1={fused_acc:.2f}")

        summary = {
            "config": args.config,
            "attributes": args.attributes,
            "checkpoint": args.checkpoint,
            "epochs": args.epochs,
            "prompt_lr": args.prompt_lr,
            "weight_lr": args.weight_lr,
            "n_ctx": args.n_ctx,
            "n_att": args.n_att,
            "max_motion": args.max_motion,
            "max_appearance": args.max_appearance,
            "fused_val_top1": fused_acc,
            "top_combinations": top_results,
        }

        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"搜索结果已保存至 {args.output}")

    # 最终同步
    if dist.is_initialized():
        dist.barrier()
    
    cleanup_distributed()


def main() -> None:
    args = parse_args()
    train_attribute_search(args)


if __name__ == "__main__":
    main()
