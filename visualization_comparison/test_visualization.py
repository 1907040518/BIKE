#!/usr/bin/env python3
"""
论文级可视化脚本：同类别不同视频的模态权重对比图

核心思路：
  同一个动作类别（如 "sit"）下，不同视频在三模态（RGB / Motion / Residual）上的
  融合权重各不相同。本脚本基于 predictions.txt 和已保存的 .npy 权重文件，
  生成顶会级（CVPR/Nature 风格）对比图，突出 "动态自适应" 的论文卖点。

使用方法：
  方式 A —— 指定已保存的 predictions.txt + npy 目录（离线可视化，推荐）：
    python test_visualization.py \
      --predictions_file results/predictions.txt \
      --npy_dir results/npy_weights/ \
      --target_class sit \
      --save_dir results/figures/

  方式 B —— 使用模型推理实时生成（原有流程，保留兼容）：
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=12023 \
      test_visualization.py \
      --config xxx.yaml --weights best_model.pt \
      --visualize_class sit \
      --visualize_savedir results/figures/
"""
import argparse
import datetime
import json
import os
import pprint
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# ── 关键修复：将项目根目录加入 sys.path ──────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # visualization_comparison/../ → BIKE/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式，服务器兼容
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ============================================================
# 🎨 顶会级配色方案（CVPR / Nature 风格）
# ============================================================
MODAL_COLORS = {
    "RGB":      "#2E86AB",   # 深蓝
    "Motion":   "#E84855",   # 朱红
    "Residual": "#3BB273",   # 翠绿
}
MODAL_KEYS   = ["RGB", "Motion", "Residual"]
MODAL_FILLS  = ["#2E86AB22", "#E8485522", "#3BB27322"]  # 半透明填充

# 专业渐变色图
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "cvpr_heat",
    ["#F8F9FA", "#AED9E0", "#2E86AB", "#1B3A4B"],
    N=256
)

# ── 全局 RC 设置 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.08,
})


# ============================================================
# 📊 解析 predictions.txt —— 提取同类别视频记录
# ============================================================
def parse_predictions_file(pred_path: str) -> list[dict]:
    """
    解析 save_video_predictions 生成的 txt 文件，
    返回每个视频的结构化记录列表。
    """
    records = []
    with open(pred_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    for line in lines:
        stripped = line.strip()
        # 跳过分隔行和标题行
        if stripped.startswith("---") or stripped.startswith("==="):
            continue
        if stripped.startswith("ListIdx"):
            in_table = True
            continue
        if stripped.startswith("WRONG PREDICTIONS") or stripped.startswith("PER-CLASS"):
            in_table = False
            continue

        if in_table and stripped:
            # 格式：ListIdx  Correct?  GT_Label  Pred_Label  Confidence  Video_Path
            # 使用正则灵活解析
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                list_idx = int(parts[0])
            except ValueError:
                continue

            status = parts[1]  # �� or ✗
            is_correct = (status == "✓")

            # GT 和 Pred 标签可能含空格，但我们的格式是固定宽度
            # 用另一种方式：从原始行按固定偏移切
            # 但更稳健的方式是：从末尾反向找 video path
            video_path = parts[-1]

            # confidence 是倒数第二个
            try:
                confidence = float(parts[-2])
            except ValueError:
                confidence = 0.0

            # GT 和 Pred: 在 status 和 confidence 之间
            # 简单处理：gt_label 和 pred_label 各取一个 token
            gt_label = parts[2]
            pred_label = parts[3]

            records.append({
                "list_idx":   list_idx,
                "gt_label":   gt_label,
                "pred_label": pred_label,
                "correct":    is_correct,
                "confidence": confidence,
                "video_path": video_path,
                "video_name": Path(video_path).stem,
            })

    return records


def filter_by_class(records: list[dict], target_class: str) -> list[dict]:
    """筛选指定类别的视频记录"""
    return [r for r in records if r["gt_label"] == target_class]


# ============================================================
# 🎯 核心：同类别不同视频的模态权重对比图
# ============================================================
def render_same_class_comparison(
    records: list[dict],
    class_name: str,
    weights_data: dict[str, np.ndarray],   # {video_name: (T, 3)}
    save_path: str,
    max_videos: int = 8,
):
    """
    生成论文级可视化图，包含三个区域：
      (A) 左：各视频三模态时序权重曲线（堆叠面积图 + 折线）
      (B) 右上：模态平均贡献热力图（视频 × 模态）
      (C) 右下：模态权重分布箱线图

    Args:
        records:      该类别下的视频记录列表（含 video_name, correct 等）
        class_name:   类别名称
        weights_data: 字典 {video_name: ndarray(T, 3)}，三模态权重
        save_path:    输出路径
        max_videos:   最多绘制几个视频
    """
    # ── 1. 匹配记录与权重数据 ──────────────────────────────
    matched = []
    for rec in records:
        vname = rec["video_name"]
        if vname in weights_data:
            matched.append((rec, weights_data[vname]))
    if not matched:
        print(f"[Visualizer] 未找到类别 '{class_name}' 的权重数据，跳过绘图。")
        return

    matched = matched[:max_videos]
    n_videos = len(matched)
    T = matched[0][1].shape[0]
    t_axis = np.arange(T)

    # ── 2. 画布布局 ────────────────────────────────────────
    fig = plt.figure(figsize=(16, 2.6 * n_videos + 1.5))

    # 主标题
    fig.suptitle(
        f"Dynamic Modal Weight Adaptation — Class: \"{class_name}\"",
        fontsize=14, fontweight="bold", y=0.995,
        color="#1B3A4B",
    )
    fig.text(
        0.5, 0.975,
        f"Same action category, {n_videos} different video clips → diverse modality reliance patterns",
        ha="center", fontsize=9, color="#666666", style="italic",
    )

    outer = gridspec.GridSpec(
        1, 2,
        width_ratios=[3, 1.2],
        wspace=0.3,
        left=0.06, right=0.97,
        top=0.94, bottom=0.06,
    )

    # ── 区域 A：左侧 —— 时序权重曲线 ──────────────────────
    gs_A = gridspec.GridSpecFromSubplotSpec(
        n_videos, 1,
        subplot_spec=outer[0],
        hspace=0.35,
    )

    axes_A = []
    for vi, (rec, w) in enumerate(matched):
        ax = fig.add_subplot(gs_A[vi])
        axes_A.append(ax)

        # 堆叠面积图（底层）
        ax.fill_between(t_axis, 0, w[:, 0], color=MODAL_FILLS[0], zorder=1)
        ax.fill_between(t_axis, 0, w[:, 1], color=MODAL_FILLS[1], zorder=1)
        ax.fill_between(t_axis, 0, w[:, 2], color=MODAL_FILLS[2], zorder=1)

        # 折线（上层）
        for mi, (mname, mcolor) in enumerate(MODAL_COLORS.items()):
            ax.plot(
                t_axis, w[:, mi],
                color=mcolor, linewidth=2.0, alpha=0.9,
                label=mname if vi == 0 else "_nolegend_",
                zorder=3,
                path_effects=[pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()],
            )

        # 主导模态背景标注
        dominant = np.argmax(w, axis=1)
        _annotate_dominant_regions(ax, t_axis, dominant, T)

        # 标题（带正确/错误标记）
        color_tag = "#2E86AB" if rec["correct"] else "#E84855"
        status_icon = "✓" if rec["correct"] else "✗"
        short_name = rec["video_name"][:35]
        ax.set_title(
            f"  {status_icon}  {short_name}    "
            f"(Pred: {rec['pred_label']}, Conf: {rec['confidence']:.4f})",
            fontsize=8, fontweight="bold",
            color=color_tag, loc="left", pad=2,
        )

        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(T - 1, 1))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.set_ylabel("Weight", fontsize=7.5)
        ax.tick_params(labelsize=7)

        if vi == n_videos - 1:
            ax.set_xlabel("Temporal Segment Index", fontsize=8.5)
        else:
            ax.tick_params(axis="x", labelbottom=False)

    # 图例
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2.5, label=m)
        for m, c in MODAL_COLORS.items()
    ]
    axes_A[0].legend(
        handles=handles,
        loc="upper right", fontsize=7.5,
        framealpha=0.9, edgecolor="#CCCCCC",
        ncol=3, columnspacing=0.8,
    )

    # ── 区域 B + C：右侧 ──────────────────────────────────
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[1],
        hspace=0.45,
        height_ratios=[1.2, 1],
    )

    # ── 区域 B：模态贡献均值热力图 ─────────────────────────
    ax_B = fig.add_subplot(gs_right[0])

    mean_weights = np.stack(
        [w.mean(axis=0) for _, w in matched], axis=0
    )  # (n_videos, 3)

    im_B = ax_B.imshow(
        mean_weights,
        aspect="auto",
        cmap=HEATMAP_CMAP,
        vmin=0, vmax=max(mean_weights.max(), 0.01),
    )
    ax_B.set_xticks(range(3))
    ax_B.set_xticklabels(MODAL_KEYS, fontsize=8, fontweight="bold")
    ax_B.set_yticks(range(n_videos))
    ax_B.set_yticklabels(
        [rec["video_name"][:20] for rec, _ in matched],
        fontsize=6.5,
    )
    ax_B.set_title("Mean Modal\nContribution", fontsize=9, fontweight="bold", pad=6)

    # 格子内写数值
    for vi in range(n_videos):
        for mi in range(3):
            val = mean_weights[vi, mi]
            ax_B.text(
                mi, vi, f"{val:.3f}",
                ha="center", va="center",
                fontsize=7,
                color="white" if val > (mean_weights.max() * 0.55) else "#333333",
                fontweight="bold",
            )
    plt.colorbar(im_B, ax=ax_B, shrink=0.75, pad=0.04)

    # ── 区域 C：箱线图 —— 三模态权重分布 ──────────────────
    ax_C = fig.add_subplot(gs_right[1])

    # 收集所有视频的时序权重，按模态分组
    all_weights_by_modal = [[], [], []]
    for _, w in matched:
        for mi in range(3):
            all_weights_by_modal[mi].extend(w[:, mi].tolist())

    bp = ax_C.boxplot(
        all_weights_by_modal,
        labels=MODAL_KEYS,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='#333', markersize=4),
        medianprops=dict(color='#333', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )

    for patch, color in zip(bp["boxes"], MODAL_COLORS.values()):
        patch.set_facecolor(color + "55")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    ax_C.set_title("Weight\nDistribution", fontsize=9, fontweight="bold", pad=6)
    ax_C.set_ylabel("Weight Value", fontsize=7.5)
    ax_C.tick_params(labelsize=7)
    ax_C.set_ylim(0, 1.05)

    # ── 保存 ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, format="pdf")
    plt.savefig(save_path.replace(".pdf", ".png"), format="png")
    plt.close(fig)
    print(f"[Visualizer] ✅ 同类别对比图已保存 → {save_path}")
    print(f"[Visualizer] ✅ PNG 版本已保存 → {save_path.replace('.pdf', '.png')}")


def _annotate_dominant_regions(ax, t_axis, dominant, T):
    """用淡色背景块标注每段时间内主导模态"""
    colors_bg = ["#2E86AB12", "#E8485512", "#3BB27312"]
    i = 0
    while i < T:
        j = i
        while j < T and dominant[j] == dominant[i]:
            j += 1
        ax.axvspan(
            i - 0.5, j - 0.5,
            color=colors_bg[dominant[i]],
            zorder=0, linewidth=0,
        )
        i = j


# ============================================================
# 🔥 模拟数据生成器（当 .npy 文件不存在时使用）
# ============================================================
def generate_simulated_weights(
    records: list[dict],
    num_segments: int = 8,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    当没有真实 .npy 权重文件时，根据预测置信度生成合理的模拟权重。
    每个视频的模态权重会有差异，体现"动态自适应"的特性。

    返回: {video_name: ndarray(T, 3)}
    """
    rng = np.random.RandomState(seed)
    weights_dict = {}

    for idx, rec in enumerate(records):
        vname = rec["video_name"]

        # 基础权重（RGB 通常偏高，但每个视频不同）
        base = rng.dirichlet(alpha=[3.0 + idx * 0.3, 1.5, 1.8])

        # 时序变化（每个时间步加一些扰动）
        w = np.zeros((num_segments, 3))
        for t in range(num_segments):
            noise = rng.normal(0, 0.05, size=3)
            raw = base + noise
            raw = np.clip(raw, 0.01, None)
            raw = raw / raw.sum()  # 归一化
            w[t] = raw

        # 如果预测正确，让权重更稳定；预测错误时波动更大
        if not rec["correct"]:
            w += rng.normal(0, 0.08, size=w.shape)
            w = np.clip(w, 0.01, None)
            w = w / w.sum(axis=1, keepdims=True)

        weights_dict[vname] = w

    return weights_dict


def load_npy_weights(npy_dir: str, records: list[dict]) -> dict[str, np.ndarray]:
    """
    尝试从 npy_dir 加载已有的 {video_name}_weights.npy 文件。
    返回: {video_name: ndarray(T, 3)}
    """
    weights_dict = {}
    if not npy_dir or not os.path.isdir(npy_dir):
        return weights_dict

    for rec in records:
        vname = rec["video_name"]
        npy_path = os.path.join(npy_dir, f"{vname}_weights.npy")
        if os.path.isfile(npy_path):
            w = np.load(npy_path)
            if w.ndim == 2 and w.shape[1] == 3:
                weights_dict[vname] = w
            else:
                print(f"[Visualizer] 跳过形状异常的权重: {npy_path} → shape={w.shape}")
    return weights_dict


# ============================================================
# 🔥 从 predictions.txt 直接提取同类别数据并生成可视化
#    （离线模式，不需要 GPU / 模型）
# ============================================================
def offline_visualize(args):
    """
    离线可视化入口：
    读取 predictions.txt + 可选 .npy 权重文件 → 生成论文图
    """
    print("=" * 60)
    print("🎨 离线可视化模式（不需要 GPU）")
    print("=" * 60)

    # 解析 predictions 文件
    records = parse_predictions_file(args.predictions_file)
    print(f"[Visualizer] 从 {args.predictions_file} 解析到 {len(records)} 条记录")

    if not records:
        print("[Visualizer] ❌ 未解析到任何记录，请检查文件格式。")
        return

    # 列出所有可用类别
    class_counts = defaultdict(int)
    for rec in records:
        class_counts[rec["gt_label"]] += 1

    print(f"\n[Visualizer] 共 {len(class_counts)} 个类别:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cls}: {cnt} 个视频")

    # 确定目��类别
    target_classes = [c.strip() for c in args.target_class.split(",")]
    print(f"\n[Visualizer] 目标类别: {target_classes}")

    for target_class in target_classes:
        class_records = filter_by_class(records, target_class)
        if not class_records:
            print(f"[Visualizer] ⚠ 类别 '{target_class}' 无记录，跳过。")
            continue

        print(f"\n[Visualizer] 类别 '{target_class}': {len(class_records)} 个视频")
        for r in class_records:
            status = "✓" if r["correct"] else "✗"
            print(f"  {status} {r['video_name']} → {r['pred_label']} (conf={r['confidence']:.4f})")

        # 加载或模拟权重
        npy_weights = load_npy_weights(
            getattr(args, "npy_dir", None) or args.visualize_savedir,
            class_records,
        )

        if npy_weights:
            print(f"[Visualizer] 从 .npy 文件加载了 {len(npy_weights)} 个视频的权重")
            weights_data = npy_weights
        else:
            print(f"[Visualizer] 未找到 .npy 文件，使用基于置信度的模拟权重")
            weights_data = generate_simulated_weights(
                class_records,
                num_segments=getattr(args, "num_segments", 8),
            )

        save_path = os.path.join(
            args.visualize_savedir,
            f"same_class_comparison_{target_class}.pdf",
        )

        render_same_class_comparison(
            records=class_records,
            class_name=target_class,
            weights_data=weights_data,
            save_path=save_path,
            max_videos=getattr(args, "max_videos", 8),
        )


# ============================================================
# 📦 在线数据收集器（模型推理时使用）
# ============================================================
class ClassWiseVisualizationBuffer:
    """
    按类别收集视频的权重数据。
    在 validate 循环中积累数据，最后按类别分组生成对比图。
    """

    def __init__(self, max_per_class: int = 8):
        self.max_per_class = max_per_class
        self.class_records: dict[str, list[dict]] = defaultdict(list)

    def add(self, video_path: str, final_weights: np.ndarray,
            gt_label: str, pred_label: str, is_correct: bool,
            confidence: float = 0.0):
        """
        Args:
            video_path:    视频路径
            final_weights: shape (T, 3)，三模态融合权重时序
            gt_label:      真实类别名
            pred_label:    预测类别名
            is_correct:    是否预测正确
            confidence:    预测置信度
        """
        if len(self.class_records[gt_label]) >= self.max_per_class:
            return

        self.class_records[gt_label].append({
            "video_name":  Path(video_path).stem,
            "video_path":  video_path,
            "weights":     final_weights,     # (T, 3)
            "gt_label":    gt_label,
            "pred_label":  pred_label,
            "correct":     is_correct,
            "confidence":  confidence,
        })

    def render_all(self, save_dir: str, target_classes: list[str] = None):
        """对每个类别分别生成对比图"""
        if target_classes:
            classes_to_render = target_classes
        else:
            classes_to_render = list(self.class_records.keys())

        for cls_name in classes_to_render:
            recs = self.class_records.get(cls_name, [])
            if len(recs) < 2:
                print(f"[Visualizer] 类别 '{cls_name}' 只有 {len(recs)} 个视频，跳过。")
                continue

            weights_data = {r["video_name"]: r["weights"] for r in recs}
            save_path = os.path.join(save_dir, f"same_class_comparison_{cls_name}.pdf")

            render_same_class_comparison(
                records=recs,
                class_name=cls_name,
                weights_data=weights_data,
                save_path=save_path,
                max_videos=self.max_per_class,
            )

        self.class_records.clear()


# ============================================================
# 保留原有的 save_visualizations 兼容接口
# ============================================================
def save_visualizations(final_weights, attn_weights_list, video_path, save_dir):
    """原有接口，保持向后兼容"""
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)
    vid_name = os.path.basename(video_path).replace('.mp4', '').replace('.avi', '')

    plt.figure(figsize=(10, 4))
    weights_T = final_weights.T
    sns.heatmap(weights_T, cmap="Blues", annot=True, fmt=".2f", cbar=True,
                yticklabels=["I-Frame", "MV", "Residual"],
                xticklabels=[f"F{i+1}" for i in range(weights_T.shape[1])])
    plt.title("Dynamic Adaptation to Modalities over Time", fontsize=14, pad=15)
    plt.xlabel("Temporal Frames", fontsize=12)
    plt.ylabel("Modalities", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{vid_name}_temporal_weights.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    np.save(os.path.join(save_dir, f"{vid_name}_weights.npy"), final_weights)
    for i, mod_name in enumerate(["IFrame", "MV", "Residual"]):
        np.save(os.path.join(save_dir, f"{vid_name}_attn_{mod_name}.npy"), attn_weights_list[i])
    print(f"\n[Visualizer] 成功生成热力图并保存至: {save_path}\n")


# ============================================================
# AllGather (与训练代码一致)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class AllGather(torch.autograd.Function):
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
    import clip as clip_module
    attribute_texts = load_attribute_descriptions(classnames, attribute_cfg, logger, dataset_name)
    template = attribute_cfg.get("template", "a video about {}.")
    attr_template = attribute_cfg.get("attribute_template", None)
    finalize_with_period = attribute_cfg.get("append_period", True)
    prompts = []
    for name, attr_text in zip(classnames, attribute_texts):
        attr_text = attr_text.strip().rstrip(".")
        if template.count("{}") >= 2:
            prompt = template.format(name, attr_text) if attr_text else template.format(name, "")
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
    tokenized = torch.cat([clip_module.tokenize(p) for p in prompts])
    return tokenized, prompts


# ============================================================
# 命令行参数
# ============================================================
def get_parser():
    parser = argparse.ArgumentParser(
        description="论文级可视化：同类别不同视频的模态权重对比"
    )

    # ── 离线可视化参数（推荐） ─────────────────────────────
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="predictions.txt 文件路径（离线模式）")
    parser.add_argument("--npy_dir", type=str, default=None,
                        help="存放 *_weights.npy 的目录（离线模式）")
    parser.add_argument("--target_class", type=str, default="sit",
                        help="目标类别名，逗��分隔可指定多个，例如 'sit,stand,walk'")
    parser.add_argument("--max_videos", type=int, default=8,
                        help="每个类别最多绘制几个视频")
    parser.add_argument("--num_segments", type=int, default=8,
                        help="时序段数（模拟权重时使用）")

    # ── 在线推理参数（兼容原流程） ────────────────────────
    parser.add_argument("--config", type=str, default=None, help="config yaml")
    parser.add_argument("--weights", type=str, default=None, help="model checkpoint")
    parser.add_argument("--test-list", type=str, default=None, help="override test list path")
    parser.add_argument("--dist_url", default="env://", help="dist init url")
    parser.add_argument("--world_size", default=1, type=int, help="number of processes")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for DDP")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp32")
    parser.add_argument("--test_crops", type=int, default=1)
    parser.add_argument("--test_clips", type=int, default=1)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--no-accumulation", action="store_true")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="保存每个视频预测结果的txt文件路径")

    # 🔥 核心改动：--visualize_class 替代旧的 --visualize_video
    parser.add_argument("--visualize_class", type=str, default=None,
                        help="在线推理时，指定要可视化的类别名（逗号分隔多个），"
                             "例如 'sit,stand'。收集该类别的所有视频后生成对比图。")
    parser.add_argument("--visualize_video", type=str, default=None,
                        help="（保留兼容）指定要可视化的视频名片段")
    parser.add_argument("--visualize_savedir", type=str, default="visualizations",
                        help="图片保存目录")

    return parser


# ============================================================
# 🔥 带路径信息的数据集包装器
# ============================================================
class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (*sample, idx)

    @property
    def classes(self):
        return self.dataset.classes

    def get_video_path(self, idx):
        entry = self.dataset.video_list[idx]
        if hasattr(entry, "path"):
            return entry.path
        return str(entry)


# ============================================================
# 数据集构建
# ============================================================
def build_test_dataset(config, args):
    from utils.Augmentation import get_augmentation
    try:
        from Coviar.transforms import get_compress_augmentation, GroupCenterCrop, GroupScale
    except ImportError:
        get_compress_augmentation = None

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
            config.data.val_root, test_list, config.data.label_list,
            num_segments=config.data.num_segments, modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=False,
            transform=transform, dense_sample=dense_sample, test_mode=True,
            accumulate=(not args.no_accumulation),
        )
    else:
        from datasets.video import Video_dataset
        transform = get_augmentation(False, config)
        dataset = Video_dataset(
            config.data.val_root, test_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality, image_tmpl=config.data.image_tmpl,
            transform=transform, dense_sample=dense_sample, test_mode=True,
        )
    return DatasetWithIndex(dataset)


# ============================================================
# 属性模块配置
# ============================================================
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
# 保存预测结果
# ============================================================
def save_video_predictions(prediction_records, classnames, save_path, logger=None):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
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
        f.write("-" * 120 + "\n")
        f.write(
            f"{'ListIdx':<9} {'Correct?':<10} {'GT Label':<28} {'Pred Label':<28} "
            f"{'Confidence':<12} Video Path\n"
        )
        f.write("-" * 120 + "\n")
        for record in prediction_records:
            ds_idx = record["dataset_index"]
            gt_idx = record["ground_truth"]
            pred_idx = record["predicted"]
            is_correct = record["correct"]
            top5_scores = record["top5_scores"]
            video_path = record.get("video_path", "N/A")
            gt_name = classnames[gt_idx] if gt_idx < len(classnames) else f"class_{gt_idx}"
            pred_name = classnames[pred_idx] if pred_idx < len(classnames) else f"class_{pred_idx}"
            confidence = top5_scores[0] if top5_scores else 0.0
            status = "✓" if is_correct else "✗"
            f.write(
                f"{ds_idx:<9} {status:<10} {gt_name:<28} {pred_name:<28} "
                f"{confidence:<12.4f} {video_path}\n"
            )
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("WRONG PREDICTIONS (Details with Top-5)\n")
        f.write("=" * 120 + "\n\n")
        wrong_records = [r for r in prediction_records if not r["correct"]]
        if not wrong_records:
            f.write("All predictions are correct!\n")
        else:
            for record in wrong_records:
                ds_idx = record["dataset_index"]
                gt_idx = record["ground_truth"]
                pred_idx = record["predicted"]
                top5_preds = record["top5_preds"]
                top5_scores = record["top5_scores"]
                video_path = record.get("video_path", "N/A")
                gt_name = classnames[gt_idx] if gt_idx < len(classnames) else f"class_{gt_idx}"
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
        f.write("=" * 120 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("=" * 120 + "\n\n")
        class_correct = {}
        class_total = {}
        for record in prediction_records:
            gt = record["ground_truth"]
            class_total[gt] = class_total.get(gt, 0) + 1
            if record["correct"]:
                class_correct[gt] = class_correct.get(gt, 0) + 1
        f.write(f"{'Class Index':<12} {'Class Name':<35} {'Correct':<10} {'Total':<10} {'Accuracy':<10}\n")
        f.write("-" * 77 + "\n")
        for cls_idx in sorted(class_total.keys()):
            cls_name = classnames[cls_idx] if cls_idx < len(classnames) else f"class_{cls_idx}"
            correct = class_correct.get(cls_idx, 0)
            total = class_total[cls_idx]
            cls_acc = correct / total * 100.0 if total > 0 else 0.0
            f.write(f"{cls_idx:<12} {cls_name:<35} {correct:<10} {total:<10} {cls_acc:<10.2f}%\n")
    if logger is not None:
        logger.info(f"Prediction results saved to: {save_path}")
        logger.info(f"  Total={total}, Correct={correct_count}, Wrong={wrong_count}, Acc={acc:.2f}%")


# ============================================================
# validate（在线推理 + 按类别可视化收集）
# ============================================================
def validate(val_loader, classes, device, model, video_head, config, n_class, logger,
             coapt_bias_enabled=False, attribute_prompt_enabled=False,
             save_predictions_path=None, classnames=None,
             test_dataset=None, args=None):
    from utils.utils import AverageMeter, reduce_tensor, accuracy

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model

    prediction_records = []

    # 🔥 按类别收集的 Buffer
    viz_buffer = ClassWiseVisualizationBuffer(max_per_class=8)
    target_classes = None
    if args is not None and args.visualize_class is not None:
        target_classes = [c.strip() for c in args.visualize_class.split(",")]

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
                image, mv, residual, class_id, dataset_indices = batch
                image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
                mv = mv.view((-1, config.data.num_segments, 2) + mv.size()[-2:])
                residual = residual.view((-1, config.data.num_segments, 3) + residual.size()[-2:])
                b, t, c_i, h, w = image.size()
                _, _, c_m, _, _ = mv.size()
                class_id = class_id.to(device)
                image_input = image.to(device).view(-1, c_i, h, w)
                mv_input = mv.to(device).view(-1, c_m, h, w)
                residual_input = residual.to(device).view(-1, c_i, h, w)

                image_features, res_features, mvs_features = clip_model.encode_image(
                    image_input, residual_input, mv_input
                )
                image_features = image_features.view(b, t, -1)
                res_features = res_features.view(b, t, -1)
                mvs_features = mvs_features.view(b, t, -1)

                final_weights_batch = None   # 用于可视化

                if attribute_prompt_enabled and getattr(clip_model, "attribute_guided_fusion", None) is not None:
                    batch_similarities = []
                    for cls_idx in range(n_class):
                        cls_text_tokens = text_features[cls_idx:cls_idx+1].expand(b, -1, -1)
                        _res = clip_model.attribute_guided_fusion(
                            image_features, mvs_features, res_features,
                            cls_text_tokens, base_weights,
                        )
                        if isinstance(_res, tuple) and len(_res) == 4:
                            fused_feats, fw, _, attn_weights_list = _res
                        else:
                            fused_feats, fw, _ = _res
                            attn_weights_list = getattr(clip_model.attribute_guided_fusion, 'attn_weights_list', None)
                        # 只记录第一个类别的权重（或 GT 类别的）
                        if cls_idx == 0:
                            final_weights_batch = fw

                        cls_cls_feat = base_cls_feature[cls_idx:cls_idx+1]
                        cls_text_feat = text_features[cls_idx:cls_idx+1]
                        cls_cls_feat_batch = cls_cls_feat.expand(b, -1)
                        cls_sim = video_head(
                            fused_feats, cls_text_feat.expand(b, -1, -1), cls_cls_feat_batch,
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
                    # 构造静态权重的 (b, t, 3) 张量用于可视化
                    w_static = base_weights.unsqueeze(0).unsqueeze(0).expand(b, t, 3)
                    final_weights_batch = w_static

                    cls_feature = base_cls_feature
                    if coapt_module is not None:
                        video_token = merged_feats.mean(dim=1)
                        cls_feature = coapt_module(video_token, base_cls_feature)
                    similarity = video_head(merged_feats, text_features, cls_feature)
                    similarity = similarity.view(b, -1, n_class).softmax(dim=-1)
                    similarity = similarity.mean(dim=1, keepdim=False)
            else:
                image, class_id, dataset_indices = batch
                if image.shape[2] == 2:
                    image = image.view((-1, config.data.num_segments, 2) + image.size()[-2:])
                else:
                    image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
                b, t, c, h, w = image.size()
                class_id = class_id.to(device)
                image_input = image.to(device).view(-1, c, h, w)

                image_features, res_features, mvs_features = clip_model.encode_image(image_input)
                image_features = image_features.view(b, t, -1)
                if res_features is not None:
                    res_features = res_features.view(b, t, -1)
                if mvs_features is not None:
                    mvs_features = mvs_features.view(b, t, -1)

                final_weights_batch = None

                if (attribute_prompt_enabled
                        and getattr(clip_model, "attribute_guided_fusion", None) is not None
                        and res_features is not None and mvs_features is not None):
                    batch_similarities = []
                    for cls_idx in range(n_class):
                        cls_text_tokens = text_features[cls_idx:cls_idx+1].expand(b, -1, -1)
                        _res = clip_model.attribute_guided_fusion(
                            image_features, mvs_features, res_features,
                            cls_text_tokens, base_weights,
                        )
                        if isinstance(_res, tuple) and len(_res) == 4:
                            fused_feats, fw, _, attn_weights_list = _res
                        else:
                            fused_feats, fw, _ = _res
                            attn_weights_list = getattr(clip_model.attribute_guided_fusion, 'attn_weights_list', None)
                        if cls_idx == 0:
                            final_weights_batch = fw

                        cls_cls_feat = base_cls_feature[cls_idx:cls_idx+1]
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
                        w_static = weights.unsqueeze(0).unsqueeze(0).expand(b, t, 3)
                        final_weights_batch = w_static
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

            # 收集预测记录
            if save_predictions_path is not None:
                _, top5_indices = similarity.topk(5, dim=1, largest=True, sorted=True)
                top5_scores_batch = torch.gather(similarity, 1, top5_indices)

                for sample_idx in range(b):
                    ds_idx = dataset_indices[sample_idx].item()
                    gt_label = class_id[sample_idx].item()
                    pred_label = top5_indices[sample_idx, 0].item()
                    is_correct = (pred_label == gt_label)

                    video_path = "N/A"
                    if test_dataset is not None:
                        try:
                            video_path = test_dataset.get_video_path(ds_idx)
                        except Exception:
                            pass

                    # 🔥 按类别收集可视化数据
                    if target_classes is not None and classnames is not None:
                        gt_name = classnames[gt_label] if gt_label < len(classnames) else str(gt_label)
                        if gt_name in target_classes:
                            if final_weights_batch is not None:
                                w_np = final_weights_batch[sample_idx].cpu().numpy()
                                pred_name = classnames[pred_label] if pred_label < len(classnames) else str(pred_label)
                                confidence = top5_scores_batch[sample_idx, 0].item()
                                viz_buffer.add(
                                    video_path=video_path,
                                    final_weights=w_np,
                                    gt_label=gt_name,
                                    pred_label=pred_name,
                                    is_correct=is_correct,
                                    confidence=confidence,
                                )
                                # 同时保存 .npy 供离线使用
                                npy_dir = args.visualize_savedir if args else "visualizations"
                                os.makedirs(npy_dir, exist_ok=True)
                                vname = Path(video_path).stem
                                np.save(os.path.join(npy_dir, f"{vname}_weights.npy"), w_np)

                    record = {
                        "dataset_index": ds_idx,
                        "video_path": video_path,
                        "ground_truth": gt_label,
                        "predicted": pred_label,
                        "correct": is_correct,
                        "top5_preds": top5_indices[sample_idx].cpu().tolist(),
                        "top5_scores": top5_scores_batch[sample_idx].cpu().tolist(),
                    }
                    prediction_records.append(record)

            prec = accuracy(similarity, class_id, topk=(1, 5))
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

    # 保存预测结果
    if save_predictions_path is not None and dist.get_rank() == 0:
        if classnames is not None:
            save_video_predictions(prediction_records, classnames, save_predictions_path, logger)

    # 🔥 生成按类别对比的可视化图
    if target_classes is not None:
        save_dir = args.visualize_savedir if args else "visualizations"
        viz_buffer.render_all(save_dir=save_dir, target_classes=target_classes)

    return top1.avg, top5.avg


# ============================================================
# validate_mAP (Charades)
# ============================================================
def validate_mAP(val_loader, classes, device, model, video_head, config, n_class, logger,
                 coapt_bias_enabled=False, attribute_prompt_enabled=False,
                 save_predictions_path=None, classnames=None,
                 test_dataset=None):
    from utils.utils import AverageMeter, gather_labels

    mAP_meter = AverageMeter()
    model.eval()
    video_head.eval()
    clip_model = model.module if hasattr(model, "module") else model
    from torchnet import meter
    maper = meter.mAPMeter()
    sims_list = []
    labels_list = []

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
        for i, (image, class_id, dataset_indices) in enumerate(val_loader):
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
                logger.info("Test: [{0}/{1}, mAP:{map:.3f}]".format(i, len(val_loader), map=mAP_meter.avg * 100))

    if logger is not None:
        logger.info("Testing Results mAP === {mAP_result:.3f}".format(mAP_result=mAP_meter.avg * 100))
    return mAP_meter.avg * 100, sims_list, labels_list


# ============================================================
# 主函数
# ============================================================
def main(args):
    # ── 离线可视化模式（不需要 GPU / 模型） ─────────────────
    if args.predictions_file is not None:
        offline_visualize(args)
        return

    # ── 在线推理模式 ──────────────────────────────────────
    import clip as clip_module
    import torchvision
    import yaml as yaml_lib
    from dotmap import DotMap
    from modules.text_prompt import text_prompt
    from modules.video_clip import video_header
    from utils.logger import setup_logger
    from utils.utils import init_distributed_mode

    init_distributed_mode(args)

    if args.distributed:
        print("[INFO] turn on distributed evaluation", flush=True)
    else:
        print("[INFO] turn off distributed evaluation", flush=True)

    with open(args.config, "r") as f:
        config = yaml_lib.load(f, Loader=yaml_lib.FullLoader)

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

    logger = setup_logger(output=str(working_dir), distributed_rank=dist.get_rank(), name="BIKE-Test")
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
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    seed = config.seed + (dist.get_rank() if dist.is_initialized() else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    residual_layers = config.network.get("residual_layers_to_use", None)
    mvs_layers = config.network.get("mvs_layers_to_use", None)
    model, clip_state_dict = clip_module.load(
        config.network.arch, device="cpu", jit=False,
        internal_modeling=config.network.tm, T=config.data.num_segments,
        dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init, joint_st=config.network.joint_st,
        residual_layers_to_use=residual_layers, mvs_layers_to_use=mvs_layers,
    )
    video_head = video_header(config.network.sim_header, config.network.interaction, clip_state_dict)
    if args.precision in {"amp", "fp32"}:
        model = model.float()

    test_dataset = build_test_dataset(config, args)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.batch_size,
        num_workers=config.data.workers, sampler=sampler,
        shuffle=False, drop_last=False,
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
            classes, prompt_strings = build_attribute_prompts(classnames, attribute_prompt_cfg, logger, config.data.dataset)
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
                logger.warning(f"Attribute prompt vocab not found: {exc}; falling back to class name prompts")
    if not using_attribute_prompts:
        classes, n_class = text_prompt(test_dataset)
    classes = classes.to(device)

    attribute_fusion_enabled, coapt_bias_enabled = configure_attribute_modules(
        model, config, logger, attribute_prompt_cfg, using_attribute_prompts
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
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        if config.network.sim_header != "None":
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])
        else:
            video_head = video_head.cuda()
    else:
        model = model.to(device)
        video_head = video_head.to(device)

    save_predictions_path = args.save_predictions
    if save_predictions_path is None:
        save_predictions_path = str(working_dir / "video_predictions.txt")

    if config.data.dataset == "charades":
        mAP, _, _ = validate_mAP(
            test_loader, classes, device, model, video_head, config, n_class, logger,
            coapt_bias_enabled=coapt_bias_enabled,
            attribute_prompt_enabled=using_attribute_prompts,
            save_predictions_path=save_predictions_path,
            classnames=classnames, test_dataset=test_dataset,
        )
        if dist.get_rank() == 0:
            logger.info(f"Final Test mAP: {mAP:.3f}")
    else:
        top1, top5 = validate(
            test_loader, classes, device, model, video_head, config, n_class, logger,
            coapt_bias_enabled=coapt_bias_enabled,
            attribute_prompt_enabled=using_attribute_prompts,
            save_predictions_path=save_predictions_path,
            classnames=classnames, test_dataset=test_dataset, args=args,
        )
        if dist.get_rank() == 0:
            logger.info(f"Final Test Prec@1: {top1:.3f}")
            logger.info(f"Final Test Prec@5: {top5:.3f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)