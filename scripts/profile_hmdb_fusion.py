#!/usr/bin/env python3
"""Benchmark HMDB51 fusion model for FLOPs and inference latency."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
from dotmap import DotMap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip
from Coviar.transforms import get_compress_augmentation
from utils.Augmentation import get_augmentation
from datasets.video_lmdb import Video_dataset as VideoLMDBDataset
from modules.video_clip import video_header
from train_comp_CoAPT_lmdb import build_attribute_prompts, CoAPTBiasAdapter
from utils.utils import AverageMeter

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:  # pragma: no cover
    FlopCountAnalysis = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile HMDB51 fusion model")
    parser.add_argument("--config", required=True, help="Path to YAML config used for training")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint (.pt) to load")
    parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--output", default="hmdb51_profile.json", help="Where to store metrics JSON")
    parser.add_argument("--no-accumulation", action="store_true", help="Disable MV/res accumulation cache")
    return parser.parse_args()


def load_config(path: str) -> DotMap:
    with open(path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    cfg = DotMap(raw_cfg)
    if "dense" not in cfg.data:
        cfg.data.dense = False
    if "shot" not in cfg.data:
        cfg.data.shot = 0
    if "evaluate" not in cfg.solver:
        cfg.solver.evaluate = False
    if "eval_freq" not in cfg.logging:
        cfg.logging.eval_freq = 1
    for split in ("train", "val"):
        iframe_key = f"iframe_{split}_path"
        mv_key = f"mv_{split}_path"
        res_key = f"res_{split}_path"
        if iframe_key not in cfg.data:
            cfg.data[iframe_key] = cfg.data.get("iframe_db_path", "")
        if mv_key not in cfg.data:
            cfg.data[mv_key] = cfg.data.get("mv_db_path", "")
        if res_key not in cfg.data:
            cfg.data[res_key] = cfg.data.get("res_db_path", "")
    return cfg


def load_classnames(label_csv: str) -> Tuple[list[str], torch.LongTensor]:
    names = []
    with open(label_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            names.append(row["name"].strip())
    tokenized = torch.cat([clip.tokenize(f"This is a video about {name}") for name in names])
    return names, tokenized


def build_attribute_tokens(classnames: list[str], default_tokens: torch.LongTensor,
                           config: DotMap) -> Tuple[torch.LongTensor, bool]:
    attribute_cfg = config.network.get("action_prompt", config.network.get("attribute_prompt", None))
    if isinstance(attribute_cfg, dict) and attribute_cfg.get("enable", False):
        try:
            tokens, _ = build_attribute_prompts(classnames, attribute_cfg, logger=None, dataset_name=config.data.dataset)
            return tokens, True
        except FileNotFoundError:
            pass
    return default_tokens, False


def build_transforms(config: DotMap):
    if config.data.modality in {"mv", "residual", "iframe"}:
        return get_compress_augmentation(False, config)
    return get_augmentation(False, config)


def build_dataset(config: DotMap, transform, accumulate: bool) -> VideoLMDBDataset:
    return VideoLMDBDataset(
        config.data.val_root,
        config.data.val_list,
        config.data.label_list,
        num_segments=config.data.num_segments,
        modality=config.data.modality,
        transform=transform,
        random_shift=False,
        test_mode=True,
        dense_sample=config.data.dense,
        num_sample=1,
        accumulate=accumulate,
        iframe_db_path=config.data.iframe_val_path,
        mv_db_path=config.data.mv_val_path,
        res_db_path=config.data.res_val_path,
        gop_size=config.data.get("GOP_SIZE", 12),
    )


def load_model(config: DotMap, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    residual_layers = config.network.get("residual_layers_to_use", None)
    mvs_layers = config.network.get("mvs_layers_to_use", None)
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
        residual_layers_to_use=residual_layers,
        mvs_layers_to_use=mvs_layers,
    )
    video_head = video_header(config.network.sim_header, config.network.interaction, clip_state)
    model = model.to(device)
    video_head = video_head.to(device)
    return model, video_head


def configure_modules(model: nn.Module, config: DotMap, attribute_prompts_enabled: bool) -> Tuple[bool, nn.Module | None]:
    attribute_cfg = config.network.get("attribute_guided_fusion", None)
    fusion_kwargs = {}
    fusion_enabled = False
    if isinstance(attribute_cfg, dict):
        fusion_enabled = bool(attribute_cfg.get("enable", attribute_prompts_enabled))
        allowed = {"hidden_dim", "num_heads", "dropout", "detach_text", "residual_scale"}
        fusion_kwargs = {k: attribute_cfg[k] for k in allowed if k in attribute_cfg}
    elif attribute_cfg is not None:
        fusion_enabled = bool(attribute_cfg)
    else:
        fusion_enabled = attribute_prompts_enabled

    if fusion_enabled and attribute_prompts_enabled:
        model.configure_attribute_guided_fusion(enable=True, **fusion_kwargs)
    else:
        model.configure_attribute_guided_fusion(enable=False)
        fusion_enabled = False

    coapt_cfg = config.network.get("coapt_bias", None)
    coapt_module = None
    if isinstance(coapt_cfg, dict):
        if coapt_cfg.get("enable", False):
            embed_dim = model.text_projection.shape[1]
            coapt_module = CoAPTBiasAdapter(embed_dim, coapt_cfg.get("layers"))
    elif coapt_cfg:
        embed_dim = model.text_projection.shape[1]
        coapt_module = CoAPTBiasAdapter(embed_dim, None)

    model.coapt_bias = coapt_module
    return fusion_enabled, coapt_module


def load_checkpoint(model: nn.Module, video_head: nn.Module, ckpt_path: str, device: torch.device) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    video_head.load_state_dict(checkpoint["fusion_model_state_dict"], strict=False)


class FusionWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, video_head: nn.Module, text_features: torch.Tensor,
                 base_cls_feature: torch.Tensor, attribute_enabled: bool, coapt_module: nn.Module | None):
        super().__init__()
        self.clip_model = clip_model
        self.video_head = video_head
        self.text_features = text_features
        self.base_cls_feature = base_cls_feature
        self.attribute_enabled = attribute_enabled
        self.coapt_module = coapt_module

    def forward(self, iframe_seq: torch.Tensor, mv_seq: torch.Tensor, residual_seq: torch.Tensor,
                prompt_tokens: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = iframe_seq.shape
        iframe_in = iframe_seq.view(-1, c, h, w)
        residual_in = residual_seq.view(-1, c, h, w)
        mv_in = mv_seq.view(-1, mv_seq.size(2), mv_seq.size(3), mv_seq.size(4))

        if self.attribute_enabled:
            fused_flat, _, _, _ = self.clip_model(iframe_in, residual_in, mv_in, prompt_tokens, return_token=True)
            merged = fused_flat.view(b, t, -1)
        else:
            image_feat, res_feat, mv_feat = self.clip_model.encode_image(iframe_in, residual_in, mv_in)
            weights = torch.softmax(self.clip_model.beta, dim=0)
            merged = (weights[0] * image_feat + weights[1] * res_feat + weights[2] * mv_feat).view(b, t, -1)

        cls_feature = self.base_cls_feature
        if self.coapt_module is not None:
            video_token = merged.mean(dim=1)
            cls_feature = self.coapt_module(video_token, self.base_cls_feature)

        logits = self.video_head(merged, self.text_features, cls_feature)
        logits = logits.view(b, -1, self.text_features.size(0)).mean(dim=1)
        return logits


def compute_gflops(wrapper: FusionWrapper, sample: Tuple[torch.Tensor, ...]) -> float | None:
    if FlopCountAnalysis is None:
        return None
    iframe_seq, mv_seq, residual_seq, prompt_tokens = sample
    try:
        flop_analyzer = FlopCountAnalysis(wrapper, (iframe_seq, mv_seq, residual_seq, prompt_tokens))
        return flop_analyzer.total() / 1e9
    except Exception as exc:
        print(f"[warn] Failed to compute FLOPs: {exc}")
        return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    classnames, default_tokens = load_classnames(cfg.data.label_list)
    prompt_tokens, attr_enabled = build_attribute_tokens(classnames, default_tokens, cfg)
    model, video_head = load_model(cfg, device)
    fusion_enabled, coapt_module = configure_modules(model, cfg, attr_enabled)
    model = model.to(device)
    load_checkpoint(model, video_head, args.checkpoint, device)
    model.eval()
    video_head.eval()

    prompt_tokens = prompt_tokens.to(device)
    base_cls_feature, text_features = model.encode_text(prompt_tokens, return_token=True)

    transform = build_transforms(cfg)
    dataset = build_dataset(cfg, transform, accumulate=(not args.no_accumulation))
    num_workers = args.num_workers if args.num_workers is not None else cfg.data.workers
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    wrapper = FusionWrapper(model, video_head, text_features, base_cls_feature, fusion_enabled, coapt_module)

    profile_loader = DataLoader(dataset, batch_size=min(args.batch_size, 2), shuffle=False, num_workers=0)
    try:
        sample_batch = next(iter(profile_loader))
    except StopIteration as exc:
        raise RuntimeError("Validation dataset is empty; cannot profile") from exc

    sample_labels = sample_batch[3].long()
    sample_prompts = prompt_tokens[sample_labels]
    sample_tensors = (
        sample_batch[0].to(device),
        sample_batch[1].to(device),
        sample_batch[2].to(device),
        sample_prompts,
    )
    gflops = compute_gflops(wrapper, sample_tensors)

    total_videos = 0
    top1 = AverageMeter()
    data_time = 0.0
    forward_time = 0.0
    iter_start = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            data_ready = time.perf_counter()
            data_time += data_ready - iter_start

            iframe_seq, mv_seq, residual_seq, labels = batch
            iframe_seq = iframe_seq.to(device, non_blocking=True)
            mv_seq = mv_seq.to(device, non_blocking=True)
            residual_seq = residual_seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_indices = labels.long()
            prompt_batch = prompt_tokens[label_indices]

            if device.type == "cuda":
                torch.cuda.synchronize()
            fwd_start = time.perf_counter()
            logits = wrapper(iframe_seq, mv_seq, residual_seq, prompt_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time += time.perf_counter() - fwd_start

            preds = logits.argmax(dim=1)
            prec1 = (preds == labels).float().mean().item() * 100.0
            top1.update(prec1, labels.size(0))
            total_videos += labels.size(0)
            iter_start = time.perf_counter()

    avg_forward = (forward_time / max(1, total_videos)) * 1000.0
    avg_total = ((data_time + forward_time) / max(1, total_videos)) * 1000.0

    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "videos": total_videos,
        "avg_forward_ms_per_video": avg_forward,
        "avg_end_to_end_ms_per_video": avg_total,
        "gflops_per_video": gflops,
        "top1": top1.avg,
        "batch_size": args.batch_size,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
