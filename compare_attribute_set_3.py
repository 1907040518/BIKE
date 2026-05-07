import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import clip

# ========== 1) 配置 ==========
file_base_attr  = "/home/neimedia/gmk/BIKE/attributes/qwen/HMDB51_1.json"  # Ours1 属性来源
file_extra_attr = "/home/neimedia/gmk/BIKE/attributes/llama3.1:8b/HMDB51_2.json"  # Ours2 属性来源
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

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
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple):
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
        "offdiag_std":  float(vals.std()),
        "top1_confusion_mean": float(row_max.mean()),
    }

def plot_heatmap(S, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(S, cmap="coolwarm", vmin=-0.2, vmax=1.0, square=True,
                xticklabels=False, yticklabels=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

def build_attr_prompts(classes, data, max_words=MAX_WORDS):
    """根据 data 字典为每个类别构建属性 prompt"""
    prompts = []
    for c in classes:
        words = normalize_attr_text(data[c], max_words=max_words)
        if len(words) == 0:
            p = f"a video of {c}"
        else:
            p = f"a video of {c}. attributes: {', '.join(words)}."
        prompts.append(p)
    return prompts

# ========== 3) 读入数据 ==========
data_1 = load_json(file_base_attr)   # HMDB51_1.json
data_4 = load_json(file_extra_attr)  # HMDB51_4.json

# 取两个文件共有的类别，保证对齐
classes = sorted(list(set(data_1.keys()) & set(data_4.keys())))
print(f"Num classes: {len(classes)}")

# ========== 4) Base 特征 ==========
base_prompts = [f"a video of {c}" for c in classes]
base_feats   = encode_texts(base_prompts)
S_base       = cosine_matrix(base_feats)

# ========== 5) Ours1 特征（HMDB51_1.json 属性，无融合） ==========
attr_prompts_1 = build_attr_prompts(classes, data_1)
attr_feats_1   = encode_texts(attr_prompts_1)
# ✅ 直接用归一化特征，不做残差融合，保留属性的真实判别力
S_ours1 = cosine_matrix(attr_feats_1)

# ========== 6) Ours2 特征（HMDB51_4.json 属性，无融合） ==========
attr_prompts_4 = build_attr_prompts(classes, data_4)
attr_feats_4   = encode_texts(attr_prompts_4)
S_ours2 = cosine_matrix(attr_feats_4)

# ========== 7) 指标对比 ==========
m_base  = offdiag_stats(S_base)
m_ours1 = offdiag_stats(S_ours1)
m_ours2 = offdiag_stats(S_ours2)

print("\n===== Inter-class Similarity Stats =====")
print(f"Base  (pure class name) : {m_base}")
print(f"Ours1 (HMDB51_1 attrs)  : {m_ours1}")
print(f"Ours2 (HMDB51_4 attrs)  : {m_ours2}")
print(f"\nΔ offdiag_mean  (Ours1 - Base) : {m_ours1['offdiag_mean']  - m_base['offdiag_mean']:+.4f}")
print(f"Δ offdiag_mean  (Ours2 - Base) : {m_ours2['offdiag_mean']  - m_base['offdiag_mean']:+.4f}")
print(f"Δ top1_confusion(Ours1 - Base) : {m_ours1['top1_confusion_mean'] - m_base['top1_confusion_mean']:+.4f}")
print(f"Δ top1_confusion(Ours2 - Base) : {m_ours2['top1_confusion_mean'] - m_base['top1_confusion_mean']:+.4f}")

# ========== 8) 可视化 ==========
os.makedirs("./outputs", exist_ok=True)
plot_heatmap(S_base,  "./outputs/heatmap_base.png")
plot_heatmap(S_ours1,  "./outputs/heatmap_ours1.png")
plot_heatmap(S_ours2, "./outputs/heatmap_ours2.png")
print("\nSaved 3 heatmaps to ./outputs/")
