import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns

# ── 项目根目录加入 sys.path ─────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import clip

# ========== 1) 配置 ==========
file_with_attr = "/home/neimedia/gmk/BIKE/attributes/qwen/HMDB51_1.json"
file_wo_attr   = "/home/neimedia/gmk/BIKE/attributes/qwen/HMDB51_4.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 你项目的 clip.load 可能需要更多参数，这里先最小化
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

# 融合系数：越大越偏向类别名，越小越偏向属性句
LAMBDA_CLASS = 0.4
MAX_WORDS = 48

# ========== 2) 工具函数 ==========
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_attr_text(attr_text, max_words=48):
    words = attr_text.lower().strip().split()
    seen, dedup = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return dedup[:max_words]

def _extract_text_feature_from_output(out):
    """
    兼容:
    1) tensor
    2) tuple: (cls_feature, token_features, ...)
    3) dict: {'text_features':...} / {'features':...}
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple):
        # 你的工程里第一个就是 base_cls_feature
        x = out[0]
        if not torch.is_tensor(x):
            raise TypeError(f"tuple[0] is not tensor: {type(x)}")
        return x
    if isinstance(out, dict):
        for k in ["text_features", "features", "cls_feature", "global_feature"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise ValueError(f"Unknown dict keys: {list(out.keys())}")
    raise TypeError(f"Unsupported encode_text output type: {type(out)}")

@torch.no_grad()
def encode_texts(text_list, batch_size=128):
    feats = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)

        # 优先走你项目接口
        try:
            out = model.encode_text(tokens, return_token=True)
        except TypeError:
            out = model.encode_text(tokens)

        x = _extract_text_feature_from_output(out)
        x = F.normalize(x, dim=-1)
        feats.append(x.cpu())
    return torch.cat(feats, dim=0)

def cosine_matrix(feats):
    return (feats @ feats.t()).numpy()

def offdiag_stats(S):
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = S[mask]
    row_max = []
    for i in range(n):
        row = S[i].copy()
        row[i] = -1e9
        row_max.append(row.max())
    row_max = np.array(row_max)
    return {
        "offdiag_mean": float(vals.mean()),
        "offdiag_std": float(vals.std()),
        "top1_confusion_mean": float(row_max.mean()),
    }

def plot_heatmap(S, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(S, cmap="coolwarm", vmin=-0.2, vmax=1.0, square=True,
                xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

# ========== 3) 读入数据 ==========
data_with = load_json(file_with_attr)
data_wo   = load_json(file_wo_attr)

classes = sorted(list(set(data_with.keys()) & set(data_wo.keys())))
print(f"Num classes: {len(classes)}")

# ========== 4) Base 特征 ==========
base_prompts = [f"a video of {c}" for c in classes]
base_feats = encode_texts(base_prompts)        # [N,D]
S_base = cosine_matrix(base_feats)

# ========== 5) Ours 特征（句级 + 残差融合，替代词平均） ==========
attr_prompts = []
for c in classes:
    words = normalize_attr_text(data_with[c], max_words=MAX_WORDS)
    if len(words) == 0:
        p = f"a video of {c}"
    else:
        # 句级属性描述，比 word-avg 更保留语义结构
        p = f"a video of {c}. attributes: {', '.join(words)}."
    attr_prompts.append(p)

attr_feats = encode_texts(attr_prompts)        # [N,D]
ours_feats = F.normalize(
    LAMBDA_CLASS * base_feats + (1.0 - LAMBDA_CLASS) * attr_feats,
    dim=-1
)
S_ours = cosine_matrix(ours_feats)

# ========== 6) 指标 ==========
m_base = offdiag_stats(S_base)
m_ours = offdiag_stats(S_ours)

print("Base:", m_base)
print("Ours:", m_ours)
print("Delta(offdiag_mean):", m_ours["offdiag_mean"] - m_base["offdiag_mean"])
print("Delta(top1_confusion_mean):", m_ours["top1_confusion_mean"] - m_base["top1_confusion_mean"])

# ========== 7) 可视化 ==========
os.makedirs("./outputs", exist_ok=True)
plot_heatmap(S_base, "Inter-class Similarity (Base)", "./outputs/heatmap_base.png")
plot_heatmap(S_ours, "Inter-class Similarity (Ours: sentence+residual fusion)", "./outputs/heatmap_ours.png")
print("Saved heatmaps to ./outputs/")
