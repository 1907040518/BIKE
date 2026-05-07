import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 📝 自定义数据区域 —— 修改这里即可
# ============================================================

# 行标签（Y轴）
row_labels = [
    "Man_Who_Cheated_Hims",
    "NOVA_ELEGANTUNIVERSE",
    "RETURN_OF_THE_KING_s"
]

# 列标签（X轴）
col_labels = ["RGB", "Motion", "Residual"]

# 数据矩阵（行 × 列，与上方标签对应）
data = np.array([
    [0.22, 0.26, 0.41],
    [0.31, 0.33, 0.36],
    [0.42, 0.33, 0.25],
])

# ============================================================
# 🎨 绘图配置区域
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

heatmap = sns.heatmap(
    data,
    ax=ax,
    annot=True,               # 显示数值
    fmt=".3f",                # 数值格式（3位小数）
    cmap="YlGnBu",            # 配色方案（可换：Blues / teal 等）
    linewidths=0,             # 格子间无分割线
    annot_kws={"size": 12, "color": "white", "weight": "bold"},
    xticklabels=col_labels,
    yticklabels=row_labels,
    cbar=True,                # 显示右侧色条
    vmin=0.0,                 # 色条最小值
    vmax=0.4,                 # 色条最大值（按需调整）
)

# ============================================================
# 🧹 去除所有文本信息，仅保留右侧色条标签
# ============================================================

ax.set_xticklabels([])        # 去掉 X 轴标签
ax.set_yticklabels([])        # 去掉 Y 轴标签
ax.set_xlabel("")             # 去掉 X 轴标题
ax.set_ylabel("")             # 去掉 Y 轴标题
ax.tick_params(left=False, bottom=False)  # 去掉刻度线

# 色条标签保留（右侧刻度数字）
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("/home/neimedia/gmk/BIKE/visualization_comparison/heatmap_output.png", dpi=150, bbox_inches="tight")
plt.show()
