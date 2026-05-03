"""
Grad-CAM Visualization for ResNet-based Compressed Video Model
==============================================================
适配 train.py 训练的 CoviarDataSet + ResNet152 模型
支持 iframe / mv / residual 三种模态

Usage:
    python /home/neimedia/gmk/BIKE/Coviar/grad_cam_resnet.py \
        --checkpoint /home/neimedia/gmk/BIKE/Coviar/hmdb51_iframe_model_iframe_model_best.pth.tar \
        --video_path '/home/neimedia/action_data/hmdb51/mpeg4_videos/turn/Veoh_Alpha_Dog_2_turn_u_nm_np2_ri_med_18.mp4' \
        --output_dir /home/neimedia/gmk/BIKE/Coviar/visualization/turn \
        --data_name hmdb51 \
        --representation iframe \
        --num_segments 16 \
        --target_class 42
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 复用你项目里已有的模块 ──────────────────────────────────
from model import Model                          # train.py 同目录
from transforms import (                  # 根据你的实际路径调整
    get_compress_augmentation,
    GroupCenterCrop,
    GroupScale,
)

try:
    from coviar import load as coviar_load
except ImportError:
    raise ImportError("请先编译安装 coviar 扩展")


# ─────────────────────────────────────────────────────────────
# 1. 经典 Grad-CAM（适用于 CNN / ResNet）
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    标准 Grad-CAM 实现。
    通过在目标卷积层注册 forward/backward hook，
    计算梯度加权的特征图激活，生成空间热力图。
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model:        已加载权重的模型（eval 模式）
            target_layer: 要挂钩的卷积层，推荐 model.base_model.layer4[-1]
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None    # 反向传播的梯度
        self.activations = None  # 前向传播的特征图

        # 注册钩子
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: [B, C, H, W]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: [B, C, H, W]
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor, target_index=None):
        """
        Args:
            input_tensor: [B*T, C, H, W]  已归一化的输入
            target_index: int，目标类别索引；None 则取 argmax

        Returns:
            cam_list: List[np.ndarray]，每帧对应一个 [H_cam, W_cam] 热力图（已归一化到 [0,1]）
        """
        self.model.zero_grad()

        # ── 前向 ──────────────────────────────────────────────
        output = self.model(input_tensor)   # [B*T, num_class]

        # 对多帧取均值，得到视频级预测
        num_segments = input_tensor.shape[0]
        output_mean = output.mean(dim=0, keepdim=True)  # [1, num_class]

        if target_index is None:
            target_index = output_mean.argmax(dim=1).item()

        # ── 反向 ──────────────────────────────────────────────
        score = output_mean[0, target_index]
        score.backward()

        # ── 计算每帧的 CAM ────────────────────────────────────
        # gradients / activations: [B*T, C, H, W]
        grads   = self.gradients    # [T, C, H, W]
        acts    = self.activations  # [T, C, H, W]

        # Global Average Pooling on spatial dims → weights [T, C]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [T, C, 1, 1]

        # 加权求和
        cam = (weights * acts).sum(dim=1)   # [T, H, W]
        cam = F.relu(cam)                   # 只保留正激活

        cam_list = []
        for t in range(cam.shape[0]):
            c = cam[t].cpu().numpy()
            c_min, c_max = c.min(), c.max()
            if c_max > c_min:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = np.zeros_like(c)
            cam_list.append(c)

        return cam_list, target_index

    def generate_smooth(self, input_tensor, target_index=None,
                        n_samples=10, noise_std=0.1):
        """SmoothGrad 版本，对噪声样本取平均，减少噪声。"""
        all_cams = []
        pred_idx = target_index

        for i in range(n_samples):
            if i == 0:
                noisy = input_tensor
            else:
                noise = torch.randn_like(input_tensor) * noise_std
                noisy = input_tensor + noise

            cam_list, pred_idx = self.generate(noisy, target_index)
            all_cams.append(cam_list)

        # 对每帧取均值
        T = len(all_cams[0])
        avg_cams = []
        for t in range(T):
            stacked = np.stack([all_cams[s][t] for s in range(n_samples)], axis=0)
            avg = stacked.mean(axis=0)
            c_min, c_max = avg.min(), avg.max()
            if c_max > c_min:
                avg = (avg - c_min) / (c_max - c_min)
            avg_cams.append(avg)

        return avg_cams, pred_idx


# ─────────────────────────────────────────────────────────────
# 2. 模型加载
# ─────────────────────────────────────────────────────────────

def load_resnet_model(checkpoint_path, num_class, num_segments,
                      representation, arch='resnet152', device='cuda'):
    """
    加载 train.py 训练的模型。
    权重保存格式：{'epoch':..., 'state_dict': OrderedDict(module.xxx)}
    """
    model = Model(num_class, num_segments, representation, base_model=arch)

    print(f"[加载权重] {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    state_dict = ckpt.get('state_dict', ckpt)

    # 去掉 DataParallel 的 'module.' 前缀
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '', 1)
        new_state[new_key] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  [警告] Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  [警告] Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    # 打印确认
    print(f"  [✓] 模型加载完成: {arch}, epoch={ckpt.get('epoch', '?')}, "
          f"best_prec1={ckpt.get('best_prec1', '?'):.2f}%")
    return model


def get_target_layer(model):
    """
    获取 ResNet152 的 Grad-CAM 目标层。
    推荐：base_model.layer4 的最后一个 Bottleneck 的最后一个 conv。
    """
    # model.base_model 是 ResNet152
    layer4 = model.base_model.layer4
    last_block = layer4[-1]          # 最后一个 Bottleneck
    target = last_block.conv3        # 最后一个 1x1 conv（输出 2048 通道）
    print(f"  [Grad-CAM 目标层] base_model.layer4[-1].conv3  "
          f"(输出通道: 2048)")
    return target


# ─────────────────────────────────────────────────────────────
# 3. 视频帧加载（复用 grad_cam.py 的逻辑）
# ─────────────────────────────────────────────────────────────

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def load_video_frames(video_path, num_segments, representation,
                      GOP_SIZE=12, accumulate=True):
    """
    使用 coviar 加载视频帧。
    返回对应 representation 的原始帧列表。
    """
    # 获取总帧数
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-count_frames',
             '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_read_frames',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=30
        )
        total_frames = int(result.stdout.strip())
    except Exception:
        total_frames = num_segments * GOP_SIZE * 2

    # 均匀采样帧索引
    seg_size = float(total_frames - 1) / num_segments
    sampled_indices = []
    for seg in range(num_segments):
        idx = int(np.round(seg_size * seg + seg_size / 2))
        idx = min(max(idx, 1), total_frames - 1)
        sampled_indices.append(idx)

    frames = []
    for frame_idx in sampled_indices:
        gop_index = frame_idx // GOP_SIZE
        gop_pos   = frame_idx % GOP_SIZE

        if representation == 'iframe':
            rep_id  = 0
            gop_pos_use = gop_pos
        elif representation == 'mv':
            rep_id  = 1
            gop_pos_use = max(gop_pos, 1)
        elif representation == 'residual':
            rep_id  = 2
            gop_pos_use = max(gop_pos, 1)
        else:
            raise ValueError(f"未知 representation: {representation}")

        try:
            frame = coviar_load(video_path, gop_index, gop_pos_use,
                                rep_id, accumulate)
            if frame is None:
                raise ValueError("coviar 返回 None")

            if representation == 'iframe':
                frame = frame[..., ::-1].copy()   # BGR → RGB
            elif representation == 'mv':
                frame = clip_and_scale(frame, 20)
                frame = (frame + 128).clip(0, 255).astype(np.uint8)
            elif representation == 'residual':
                frame = (frame + 128).clip(0, 255).astype(np.uint8)

        except Exception as e:
            print(f"  [警告] 帧 {frame_idx} 加载失败: {e}")
            if representation == 'mv':
                frame = np.zeros((256, 256, 2), dtype=np.uint8)
            else:
                frame = np.zeros((256, 256, 3), dtype=np.uint8)

        frames.append(frame)

    return frames, sampled_indices


def preprocess_frames(frames, representation, crop_size=224):
    """
    将原始帧列表预处理为模型输入 Tensor。
    复用 train.py 的 val 变换：GroupScale → GroupCenterCrop。
    """
    scale_size = int(crop_size * 256 / 224)   # 通常 256

    processed = []
    for frame in frames:
        img = frame.copy()

        # 确保是 uint8 HxWxC
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Resize
        h, w = img.shape[:2]
        if h < w:
            new_h = scale_size
            new_w = int(w * scale_size / h)
        else:
            new_w = scale_size
            new_h = int(h * scale_size / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center Crop
        top  = (new_h - crop_size) // 2
        left = (new_w - crop_size) // 2
        img  = img[top:top+crop_size, left:left+crop_size]

        # HxWxC → CxHxW，归一化到 [0, 1]
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)

        # mv 是 2 通道，需要特殊处理
        img_f = img.astype(np.float32) / 255.0
        processed.append(img_f)

    # [T, H, W, C] → [T, C, H, W]
    arr = np.stack(processed, axis=0)
    arr = np.transpose(arr, (0, 3, 1, 2))
    tensor = torch.from_numpy(arr).float()
    return tensor


# ─────────────────────────────────────────────────────────────
# 4. 可视化工具（与 grad_cam.py 保持风格一致）
# ─────────────────────────────────────────────────────────────

def apply_colormap(org_img, cam, colormap='jet', alpha=0.5):
    """将 CAM 热力图叠加到原始图像上。"""
    H, W = org_img.shape[:2]
    heatmap = cv2.resize(cam.astype(np.float32), (W, H),
                         interpolation=cv2.INTER_CUBIC)
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    superimposed = (alpha * heatmap_rgb + (1 - alpha) * org_img).astype(np.uint8)
    return superimposed, heatmap


def mv_to_rgb(mv_frame):
    """运动向量 2ch → HSV → RGB，便于可视化。"""
    flow = mv_frame.astype(np.float32) - 128.0
    mag   = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hsv   = np.zeros((*mv_frame.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag / max(mag.max(), 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_display_image(frame, representation):
    """获取用于显示的 RGB 图像。"""
    if representation == 'mv' and frame.shape[-1] == 2:
        return mv_to_rgb(frame)
    img = frame.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_gradcam_results(cam_list, original_frames, representation,
                         output_dir, target_class, use_smooth=True,
                         grid_cols=4):
    """
    保存每帧的热力图，并生成网格总览图。
    输出结构与 grad_cam.py 完全一致，方便直接对比。
    """
    os.makedirs(output_dir, exist_ok=True)
    num_segments = len(cam_list)

    # ── 保存单帧图像 ──────────────────────────────────────────
    for t, (cam, frame) in enumerate(zip(cam_list, original_frames)):
        display_img = get_display_image(frame, representation)

        superimposed, heatmap = apply_colormap(display_img, cam)

        cv2.imwrite(
            os.path.join(output_dir, f'frame_{t:03d}_original.jpg'),
            cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(output_dir, f'frame_{t:03d}_attention.jpg'),
            cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
        )
        heatmap_vis = (plt.get_cmap('jet')(heatmap)[:, :, :3] * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, f'frame_{t:03d}_heatmap.jpg'),
            cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)
        )

    # ── 网格总览图（与 grad_cam.py 完全一致的布局）────────────
    grid_rows = (num_segments + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows * 2, grid_cols,
                             figsize=(grid_cols * 4, grid_rows * 8))
    if grid_rows * 2 == 1:
        axes = axes[np.newaxis, :]
    if grid_cols == 1:
        axes = axes[:, np.newaxis]

    for t in range(num_segments):
        row = (t // grid_cols) * 2
        col = t % grid_cols
        display_img = get_display_image(original_frames[t], representation)

        axes[row, col].imshow(display_img)
        axes[row, col].set_title(f'{representation} Frame {t}', fontsize=10)
        axes[row, col].axis('off')

        superimposed, _ = apply_colormap(display_img, cam_list[t])
        axes[row+1, col].imshow(superimposed)
        axes[row+1, col].set_title(f'Grad-CAM Frame {t}', fontsize=10)
        axes[row+1, col].axis('off')

    # 隐藏多余格子
    for t in range(num_segments, grid_rows * grid_cols):
        row = (t // grid_cols) * 2
        col = t % grid_cols
        axes[row, col].axis('off')
        axes[row+1, col].axis('off')

    plt.suptitle(
        f'ResNet Grad-CAM: {representation} (class={target_class})',
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'gradcam_grid_{representation}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

    # ── 平均热力图 ────────────────────────────────────────────
    avg_cam = np.mean(cam_list, axis=0)
    avg_cam = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(avg_cam, cmap='jet', interpolation='bilinear')
    ax.set_title(f'Averaged Grad-CAM: {representation}', fontsize=14)
    ax.axis('off')
    plt.savefig(
        os.path.join(output_dir, f'gradcam_averaged_{representation}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

    print(f"  [✓] {representation} Grad-CAM 结果已保存至 {output_dir}")
    return cam_list


# ─────────────────────────────────────────────────────────────
# 5. 主函数
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet Grad-CAM 可视化')
    parser.add_argument('--checkpoint',    required=True,
                        help='模型权重路径 (.pth.tar)')
    parser.add_argument('--video_path',    required=True,
                        help='输入视频路径 (.mp4)')
    parser.add_argument('--output_dir',    default='gradcam_resnet_output',
                        help='输出目录')
    parser.add_argument('--data_name',     default='hmdb51',
                        choices=['hmdb51', 'ucf101'])
    parser.add_argument('--representation',default='iframe',
                        choices=['iframe', 'mv', 'residual'])
    parser.add_argument('--arch',          default='resnet152')
    parser.add_argument('--num_segments',  type=int, default=3)
    parser.add_argument('--target_class',  type=int, default=None,
                        help='目标类别索引；不指定则取模型预测类')
    parser.add_argument('--use_smooth',    action='store_true', default=True,
                        help='使用 SmoothGrad 减少噪声')
    parser.add_argument('--smooth_samples',type=int, default=10)
    parser.add_argument('--smooth_std',    type=float, default=0.1)
    parser.add_argument('--crop_size',     type=int, default=224)
    parser.add_argument('--no_accumulate', action='store_true', default=False)
    parser.add_argument('--device',        default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()

    # 数据集类别数
    num_class = 51 if args.data_name == 'hmdb51' else 101
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    accumulate = not args.no_accumulate

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载模型 ───────────────────────────────────────────
    model = load_resnet_model(
        args.checkpoint, num_class, args.num_segments,
        args.representation, args.arch, device
    )

    # ── 2. 设置 Grad-CAM 目标层 ───────────────────────────────
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # ── 3. 加载视频帧 ─────────────────────────────────────────
    print(f"\n[加载视频] {args.video_path}")
    frames, indices = load_video_frames(
        args.video_path, args.num_segments,
        args.representation, accumulate=accumulate
    )
    print(f"  采样帧索引: {indices}")

    # ── 4. 预处理 ─────────────────────────────────────────────
    input_tensor = preprocess_frames(
        frames, args.representation, args.crop_size
    ).to(device)
    print(f"  输入 Tensor shape: {input_tensor.shape}")

    # ── 5. 生成 Grad-CAM ──────────────────────────────────────
    print(f"\n[生成 Grad-CAM] use_smooth={args.use_smooth}")
    if args.use_smooth:
        cam_list, pred_class = gradcam.generate_smooth(
            input_tensor,
            target_index=args.target_class,
            n_samples=args.smooth_samples,
            noise_std=args.smooth_std
        )
    else:
        cam_list, pred_class = gradcam.generate(
            input_tensor, target_index=args.target_class
        )

    print(f"  预测/目标类别: {pred_class}")
    for t, cam in enumerate(cam_list):
        print(f"  Frame {t}: cam range [{cam.min():.4f}, {cam.max():.4f}]")

    # ── 6. 保存结果 ───────────────────────────────────────────
    mod_dir = os.path.join(args.output_dir, args.representation)
    save_gradcam_results(
        cam_list, frames, args.representation,
        mod_dir, pred_class,
        use_smooth=args.use_smooth
    )

    # 清理钩子
    gradcam.remove_hooks()
    print(f"\n[✓] 全部完成！结果保存至 {args.output_dir}")


if __name__ == '__main__':
    main()
