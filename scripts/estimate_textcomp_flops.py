#!/usr/bin/env python3
"""Analytical FLOPs estimator for TextComp / BIKE paper tables.

This script does not run a model forward pass. It estimates MACs-style FLOPs
from the architecture used by train_comp_CoAPT.py: CLIP ViT-B/16 visual stream,
lightweight residual/MV encoders, Instance-Aware Dynamic Fusion, and the
6-layer temporal video head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ViTSpec:
    image_size: int = 224
    patch_size: int = 16
    width: int = 768
    embed_dim: int = 512
    layers: int = 12
    heads: int = 12
    in_chans: int = 3


@dataclass
class TextCompSpec:
    num_segments: int = 16
    num_classes: int = 174
    residual_layers: int = 2
    mvs_layers: int = 2
    temporal_layers: int = 6
    temporal_width: int = 512
    fusion_hidden_dim: int = 256
    num_modalities: int = 3
    num_attributes: int = 16
    views_spatial: int = 1
    views_temporal: int = 1


def linear_macs(tokens: int, in_dim: int, out_dim: int) -> int:
    return tokens * in_dim * out_dim


def conv2d_macs(out_h: int, out_w: int, out_ch: int, in_ch: int, k_h: int, k_w: int) -> int:
    return out_h * out_w * out_ch * in_ch * k_h * k_w


def transformer_block_macs(tokens: int, width: int, mlp_ratio: int = 4) -> int:
    # qkv + output projection
    qkv_proj = 3 * linear_macs(tokens, width, width)
    out_proj = linear_macs(tokens, width, width)
    # QK^T and AV. Softmax/norm/activation are intentionally ignored, as common FLOPs tables do.
    attn = 2 * tokens * tokens * width
    # MLP: c_fc + c_proj
    mlp = linear_macs(tokens, width, width * mlp_ratio) + linear_macs(tokens, width * mlp_ratio, width)
    return qkv_proj + out_proj + attn + mlp


def visual_vit_macs(spec: ViTSpec, layers: int | None = None, include_conv_2to3: bool = False) -> int:
    layers = spec.layers if layers is None else layers
    grid = spec.image_size // spec.patch_size
    tokens = grid * grid + 1
    macs = conv2d_macs(grid, grid, spec.width, spec.in_chans, spec.patch_size, spec.patch_size)
    if include_conv_2to3:
        macs += conv2d_macs(spec.image_size, spec.image_size, 3, 2, 1, 1)
    macs += layers * transformer_block_macs(tokens, spec.width)
    macs += spec.width * spec.embed_dim  # final CLS projection
    return macs


def temporal_head_macs(num_segments: int, width: int = 512, layers: int = 6) -> int:
    return layers * transformer_block_macs(num_segments, width)


def fusion_macs(num_segments: int, hidden_dim: int, embed_dim: int, num_modalities: int, num_attributes: int) -> int:
    t, h, d, a, m = num_segments, hidden_dim, embed_dim, num_attributes, num_modalities
    macs = 0
    # K/V projections from class text/attribute tokens.
    macs += 2 * a * d * h
    for _ in range(m):
        macs += t * d * h          # q_proj
        macs += t * a * h          # QK^T
        macs += t * a * h          # Attn @ V
        macs += t * h * d          # feat_inject_proj
    # dynamic_router: Linear(m*h+d -> h) + Linear(h -> m)
    macs += t * ((m * h + d) * h + h * m)
    # weighted sum is tiny but included.
    macs += t * m * d
    return macs


def read_config_defaults(path: str | None, spec: TextCompSpec, vit: ViTSpec) -> None:
    if not path or yaml is None:
        return
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {})
    net = cfg.get("network", {})
    spec.num_segments = int(data.get("num_segments", spec.num_segments))
    spec.num_classes = int(data.get("num_classes", spec.num_classes))
    vit.image_size = int(data.get("input_size", vit.image_size))
    spec.residual_layers = len(net.get("residual_layers_to_use", [0, 11]))
    spec.mvs_layers = len(net.get("mvs_layers_to_use", [0, 11]))
    spec.temporal_layers = 6 if net.get("sim_header", "Transf") == "Transf" else 0
    fusion_cfg = net.get("attribute_guided_fusion", {}) or {}
    spec.fusion_hidden_dim = int(fusion_cfg.get("hidden_dim", spec.fusion_hidden_dim))
    action_cfg = net.get("action_prompt", {}) or {}
    spec.num_attributes = int(action_cfg.get("num_attributes", spec.num_attributes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate TextComp FLOPs for paper tables.")
    parser.add_argument("--config", default="configs/sthv2/sthv2_pre_fix_B16.yaml")
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--spatial-crops", type=int, default=1)
    parser.add_argument("--temporal-clips", type=int, default=1)
    parser.add_argument("--class-conditioned", action="store_true", help="Count fusion+temporal head once per class, matching current validation logits.")
    args = parser.parse_args()

    vit = ViTSpec()
    spec = TextCompSpec(views_spatial=args.spatial_crops, views_temporal=args.temporal_clips)
    read_config_defaults(args.config, spec, vit)
    if args.segments is not None:
        spec.num_segments = args.segments
    if args.classes is not None:
        spec.num_classes = args.classes

    image_per_frame = visual_vit_macs(vit, layers=vit.layers)
    residual_per_frame = visual_vit_macs(vit, layers=spec.residual_layers)
    mvs_per_frame = visual_vit_macs(vit, layers=spec.mvs_layers, include_conv_2to3=True)

    visual_per_view = spec.num_segments * (image_per_frame + residual_per_frame + mvs_per_frame)
    fusion_once = fusion_macs(spec.num_segments, spec.fusion_hidden_dim, vit.embed_dim, spec.num_modalities, spec.num_attributes)
    temporal_once = temporal_head_macs(spec.num_segments, spec.temporal_width, spec.temporal_layers)
    classifier_once = spec.num_classes * vit.embed_dim

    multiplier = spec.num_classes if args.class_conditioned else 1
    per_view = visual_per_view + multiplier * (fusion_once + temporal_once + classifier_once)
    total_views = spec.views_spatial * spec.views_temporal
    total = per_view * total_views

    def g(x: int) -> float:
        return x / 1e9

    print("TextComp FLOPs estimate (MACs-style, 1 multiply-add = 1 operation)")
    print(f"config: {args.config}")
    print(f"views: {spec.num_segments}x{spec.views_spatial}x{spec.views_temporal} = {total_views} inference view(s), {spec.num_segments} segments per view")
    print(f"num_classes: {spec.num_classes}")
    print(f"class_conditioned: {args.class_conditioned}")
    print(f"image_stream_per_frame_G: {g(image_per_frame):.3f}")
    print(f"residual_stream_per_frame_G: {g(residual_per_frame):.3f}")
    print(f"mvs_stream_per_frame_G: {g(mvs_per_frame):.3f}")
    print(f"visual_per_view_G: {g(visual_per_view):.3f}")
    print(f"fusion_once_G: {g(fusion_once):.3f}")
    print(f"temporal_head_once_G: {g(temporal_once):.3f}")
    print(f"per_view_total_G: {g(per_view):.3f}")
    print(f"total_inference_G: {g(total):.3f}")


if __name__ == "__main__":
    main()
