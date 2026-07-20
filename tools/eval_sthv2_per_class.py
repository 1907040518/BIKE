#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from contextlib import suppress

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from dotmap import DotMap

# Make local BIKE imports work when executed from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip
from modules.video_clip import video_header
from Coviar.transforms import get_compress_augmentation
from datasets.video3 import Video_dataset
# Import helper functions from the current clean training path.
from train_comp_CoAPT import (
    _get_encoder_layers_to_use,
    _get_cfg_value,
    build_attribute_prompts,
    compute_class_conditioned_logits,
    update_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SSv2 checkpoint and export per-class accuracy.")
    parser.add_argument("--config", required=True, help="Path to training yaml config")
    parser.add_argument("--weights", required=True, help="Path to model_best.pt or last_model.pt")
    parser.add_argument("--output-dir", required=True, help="Directory for csv outputs")
    parser.add_argument("--batch-size", type=int, default=16, help="Validation batch size for single-GPU eval")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    parser.add_argument("--class-chunk-size", type=int, default=None, help="Override solver.class_chunk_size")
    parser.add_argument("--no-accumulation", action="store_true", help="Disable accumulation of motion vectors/residuals")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only: stop after N batches")
    return parser.parse_args()


def load_labels(label_csv):
    labels = []
    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append((int(row["id"]), row["name"]))
    labels.sort(key=lambda x: x[0])
    return labels


def build_model_and_data(config, args, device):
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
        residual_layers_to_use=_get_encoder_layers_to_use(config, "residual_layers_to_use", [0, 1]),
        mvs_layers_to_use=_get_encoder_layers_to_use(config, "mvs_layers_to_use", [0, 1]),
    )
    video_head = video_header(config.network.sim_header, config.network.interaction, clip_state_dict)
    model = model.float().to(device)
    video_head = video_head.float().to(device)

    transform_val = get_compress_augmentation(False, config)
    val_data = Video_dataset(
        config.data.val_root,
        config.data.val_list,
        config.data.label_list,
        random_shift=False,
        num_segments=config.data.num_segments,
        modality=config.data.modality,
        test_mode=True,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val,
        dense_sample=config.data.dense,
        accumulate=(not args.no_accumulation),
    )

    classnames = [c for _, c in val_data.classes]
    attribute_prompt_cfg = config.network.get("action_prompt", config.network.get("attribute_prompt", None))
    if not (isinstance(attribute_prompt_cfg, dict) and attribute_prompt_cfg.get("enable", False)):
        raise RuntimeError("This evaluator expects action_prompt/attribute_prompt to be enabled for the clean CoAPT path.")
    classes, _ = build_attribute_prompts(classnames, attribute_prompt_cfg, logger=None, dataset_name=config.data.dataset)
    classes = classes.to(device)

    attribute_fusion_cfg = config.network.get("attribute_guided_fusion", None)
    fusion_kwargs = {}
    if isinstance(attribute_fusion_cfg, dict) and bool(attribute_fusion_cfg.get("enable", True)):
        allowed_keys = {"hidden_dim", "num_heads", "dropout", "detach_text"}
        fusion_kwargs = {k: attribute_fusion_cfg[k] for k in allowed_keys if k in attribute_fusion_cfg}
        model.configure_attribute_guided_fusion(enable=True, **fusion_kwargs)
    else:
        model.configure_attribute_guided_fusion(enable=False)

    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(update_dict(ckpt["model_state_dict"]), strict=False)
    video_head.load_state_dict(update_dict(ckpt["fusion_model_state_dict"]), strict=False)
    model = model.to(device)
    video_head = video_head.to(device)
    print(f"Loaded checkpoint: {args.weights} epoch={ckpt.get('epoch', 'N/A')}", flush=True)

    loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    return model, video_head, loader, classes, classnames


@torch.no_grad()
def evaluate(config, args, model, video_head, loader, classes, classnames, device):
    model.eval()
    video_head.eval()
    autocast = torch.cuda.amp.autocast if args.precision == "amp" and device.type == "cuda" else suppress

    base_cls_feature, text_features = model.encode_text(classes, return_token=True)
    base_weights = F.softmax(model.beta, dim=0)
    n_class = len(classnames)
    class_chunk_size = args.class_chunk_size or _get_cfg_value(config.solver, "class_chunk_size", 32)

    total = torch.zeros(n_class, dtype=torch.long)
    correct = torch.zeros(n_class, dtype=torch.long)
    top5_correct = torch.zeros(n_class, dtype=torch.long)
    rows = []

    for i, batch in enumerate(loader):
        if len(batch) == 5:
            image, mv, residual, class_id, sample_name = batch
        else:
            image, mv, residual, class_id = batch
            sample_name = [""] * len(class_id)

        image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
        mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
        residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])
        b, t, c_i, h, w = image.size()
        _, _, c_m, _, _ = mv.size()

        class_id = class_id.to(device, non_blocking=True)
        image_input = image.to(device, non_blocking=True).view(-1, c_i, h, w)
        mv_input = mv.to(device, non_blocking=True).view(-1, c_m, h, w)
        residual_input = residual.to(device, non_blocking=True).view(-1, c_i, h, w)

        with autocast():
            image_features, res_features, mvs_features = model.encode_image(image_input, residual_input, mv_input)
            image_features = image_features.view(b, t, -1)
            res_features = res_features.view(b, t, -1)
            mvs_features = mvs_features.view(b, t, -1)
            if getattr(model, "attribute_guided_fusion", None) is not None:
                similarity = compute_class_conditioned_logits(
                    clip_model=model,
                    video_head=video_head,
                    image_features=image_features,
                    residual_features=res_features,
                    mvs_features=mvs_features,
                    base_cls_feature=base_cls_feature,
                    text_features=text_features,
                    base_weights=base_weights,
                    n_class=n_class,
                    chunk_size=class_chunk_size,
                )
                similarity = model.logit_scale.exp() * similarity
            else:
                merged_feats = base_weights[0] * image_features + base_weights[1] * res_features + base_weights[2] * mvs_features
                similarity = video_head(merged_feats, text_features, base_cls_feature)
                similarity = similarity.view(b, -1, n_class).softmax(dim=-1).mean(dim=1)

        pred = similarity.argmax(dim=1)
        top5 = similarity.topk(min(5, n_class), dim=1).indices
        is_correct = pred.eq(class_id)
        is_top5 = top5.eq(class_id.view(-1, 1)).any(dim=1)

        for gt, pd, ok, ok5, name in zip(class_id.cpu().tolist(), pred.cpu().tolist(), is_correct.cpu().tolist(), is_top5.cpu().tolist(), sample_name):
            total[gt] += 1
            correct[gt] += int(ok)
            top5_correct[gt] += int(ok5)
            rows.append([name, gt, classnames[gt], pd, classnames[pd], int(ok), int(ok5)])

        if i % 10 == 0:
            seen = int(total.sum())
            acc = 100.0 * int(correct.sum()) / max(1, seen)
            print(f"[{i}/{len(loader)}] seen={seen} top1={acc:.3f}", flush=True)
        if args.max_batches is not None and i + 1 >= args.max_batches:
            break

    return total, correct, top5_correct, rows


def select_middle_classes(total, correct, classnames, k=11):
    stats = []
    for idx, name in enumerate(classnames):
        if total[idx].item() <= 0:
            continue
        acc = correct[idx].item() / total[idx].item()
        stats.append((idx, name, acc, int(correct[idx]), int(total[idx])))
    stats.sort(key=lambda x: (x[2], x[4], x[0]))
    mid = len(stats) // 2
    half = k // 2
    start = max(0, mid - half)
    end = min(len(stats), start + k)
    start = max(0, end - k)
    selected = stats[start:end]
    selected_ids = {x[0] for x in selected}
    return stats, selected, selected_ids


def filter_list(src, dst, selected_ids):
    kept = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[-1])
            if label in selected_ids:
                fout.write(line)
                kept += 1
    return kept


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(raw_config)
    if args.class_chunk_size is not None:
        config.solver.class_chunk_size = args.class_chunk_size

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, video_head, loader, classes, classnames = build_model_and_data(config, args, device)
    total, correct, top5_correct, pred_rows = evaluate(config, args, model, video_head, loader, classes, classnames, device)

    stats, selected, selected_ids = select_middle_classes(total, correct, classnames, k=11)

    per_class_csv = output_dir / "sthv2_per_class_acc.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank_by_acc_low_to_high", "class_id", "class_name", "top1_acc", "top5_acc", "correct", "top5_correct", "total", "selected_middle11"])
        for rank, (idx, name, acc, corr, tot) in enumerate(stats):
            top5_acc = top5_correct[idx].item() / max(1, total[idx].item())
            writer.writerow([rank, idx, name, f"{acc:.6f}", f"{top5_acc:.6f}", corr, int(top5_correct[idx]), tot, int(idx in selected_ids)])

    pred_csv = output_dir / "sthv2_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "gt_id", "gt_name", "pred_id", "pred_name", "top1_correct", "top5_correct"])
        writer.writerows(pred_rows)

    selected_csv = output_dir / "sthv2_middle11_classes.csv"
    with open(selected_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "top1_acc", "correct", "total"])
        for idx, name, acc, corr, tot in selected:
            writer.writerow([idx, name, f"{acc:.6f}", corr, tot])

    subset_dir = output_dir / "lists"
    subset_dir.mkdir(exist_ok=True)
    train_out = subset_dir / "train_rgb_middle11_fullsamples.txt"
    val_out = subset_dir / "val_rgb_middle11_fullsamples.txt"
    train_kept = filter_list(config.data.train_list, train_out, selected_ids)
    val_kept = filter_list(config.data.val_list, val_out, selected_ids)

    print("\nSelected middle-11 classes:", flush=True)
    for idx, name, acc, corr, tot in selected:
        print(f"  {idx:3d} | {acc*100:6.2f}% | {corr:4d}/{tot:<4d} | {name}", flush=True)
    print(f"\nSaved: {per_class_csv}", flush=True)
    print(f"Saved: {selected_csv}", flush=True)
    print(f"Saved: {pred_csv}", flush=True)
    print(f"Saved subset train list: {train_out} ({train_kept} samples)", flush=True)
    print(f"Saved subset val list:   {val_out} ({val_kept} samples)", flush=True)


if __name__ == "__main__":
    main()
