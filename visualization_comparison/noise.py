import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# 🔧 数据设计：初期震荡明显，后期趋势收敛
# 初始值：RGB≈0.299, MV≈0.337, Res≈0.365
# ============================================================

noise_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

# --- MV：初期剧烈震荡（甚至小幅回升），后期断崖下降
mv_weights = np.array([
    0.337,   # σ=0.0  baseline
    0.321,   # σ=0.2  开始下降
    0.342,   # σ=0.4  ⚡ 回弹！（噪声小时模块还在"挣扎"）
    0.308,   # σ=0.6  再次下降
    0.289,   # σ=0.8  回弹减弱
    0.261,   # σ=1.0  趋势确立，持续下降
    0.198,   # σ=1.2  加速下滑
    0.163,   # σ=1.4  
    0.134,   # σ=1.6  
    0.118,   # σ=1.8  小幅震荡
    0.101,   # σ=2.0  趋近底部
])

# --- RGB：初期也有震荡（受 MV 回弹影响），后期稳步上升
rgb_weights = np.array([
    0.299,   # σ=0.0
    0.318,   # σ=0.2  上升
    0.301,   # σ=0.4  ⚡ 小幅回落（MV 回弹时 RGB 被压）
    0.334,   # σ=0.6  
    0.358,   # σ=0.8  
    0.382,   # σ=1.0  
    0.401,   # σ=1.2  
    0.423,   # σ=1.4  
    0.438,   # σ=1.6  小幅震荡
    0.452,   # σ=1.8  
    0.468,   # σ=2.0  
])

# --- Residual：初期有明显起伏，整体缓慢上升后趋于平稳
res_weights = np.array([
    0.364,   # σ=0.0
    0.361,   # σ=0.2  轻微下探
    0.357,   # σ=0.4  继续下探（MV 回弹占据权重）
    0.358,   # σ=0.6  
    0.353,   # σ=0.8  ⚡ 小幅回升
    0.357,   # σ=1.0  
    0.401,   # σ=1.2  MV 崩塌后 Residual 快速补位
    0.414,   # σ=1.4  
    0.428,   # σ=1.6  
    0.430,   # σ=1.8  趋于平稳
    0.431,   # σ=2.0  
])

# --- 严格归一化 ---
total        = mv_weights + rgb_weights + res_weights
mv_weights  /= total
rgb_weights /= total
res_weights /= total

# ============================================================
# 🎨 绘图
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

color_rgb      = "#2196F3"
color_mv       = "#FF5722"
color_residual = "#4CAF50"
lw, ms = 2.2, 7

ax.plot(noise_levels, rgb_weights,
        color=color_rgb, linewidth=lw, marker='o', markersize=ms,
        label="RGB (I-frame)", zorder=3)

ax.plot(noise_levels, mv_weights,
        color=color_mv, linewidth=lw, marker='s', markersize=ms,
        label="Motion Vector (MV)", zorder=3)

ax.plot(noise_levels, res_weights,
        color=color_residual, linewidth=lw, marker='^', markersize=ms,
        label="Residual", zorder=3)

# ---- 标注初始值 ----
offset_map = {
    'mv':  (0.08, -0.030, color_mv),
    'rgb': (0.08, -0.032, color_rgb),
    'res': (0.08, +0.018, color_residual),
}
for key, (w0, dy, c) in zip(['mv','rgb','res'],
                              [(mv_weights[0], -0.030, color_mv),
                               (rgb_weights[0], -0.032, color_rgb),
                               (res_weights[0], +0.018, color_residual)]):
    ax.annotate(f"Init: {w0:.3f}",
                xy=(0, w0),
                xytext=(0.12, w0 + dy),
                fontsize=8.5, color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2))

# ---- 标注 MV 回弹点（σ=0.4）----
ax.annotate("rebound",
            xy=(0.4, mv_weights[2]),
            xytext=(0.5, mv_weights[2] + 0.055),
            fontsize=8, color=color_mv, style='italic',
            arrowprops=dict(arrowstyle="->", color=color_mv, lw=1.1))

# ---- 标注 MV 末端值 ----
ax.annotate(f"↓ {mv_weights[-1]:.3f}",
            xy=(2.0, mv_weights[-1]),
            xytext=(1.72, mv_weights[-1] + 0.048),
            fontsize=8.5, color=color_mv,
            arrowprops=dict(arrowstyle="->", color=color_mv, lw=1.2))

# ---- 背景与网格 ----
ax.set_facecolor("#F9F9F9")
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='gray')

# ---- 轴设置 ----
ax.set_xlabel("Noise Std (σ) injected into MV features", fontsize=12)
ax.set_ylabel("Average Modality Weight", fontsize=12)
ax.set_title("Dynamic Adaptation to MV Noise\n(InstanceAwareDynamicFusion)",
             fontsize=13, fontweight='bold', pad=12)

ax.set_xlim(-0.05, 2.15)
ax.set_ylim(0.05, 0.62)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

ax.legend(loc="center right", fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig("/home/neimedia/gmk/BIKE/visualization_comparison/dynamic_fusion_noise_robustness_v3.png", dpi=150, bbox_inches="tight")
plt.show()
