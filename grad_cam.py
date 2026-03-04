"""
Grad-CAM Visualization for Compressed Video (iframe / residual / mv)
====================================================================
V2 Fix: Uses input-gradient based attention visualization that works
correctly with gradient checkpointing (torch.utils.checkpoint).

The key insight: instead of hooking *inside* the transformer (which
checkpoint breaks), we compute gradients of the classification score
w.r.t. the **patch embeddings** that enter the transformer. These
embeddings are always on the main autograd graph.

Usage:
    python grad_cam_fixed_v2.py \
        --config configs/k400_res.yaml \
        --checkpoint model_best.pt \
        --video_path /path/to/video.mp4 \
        --output_dir gradcam_output \
        --modality all
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
import yaml
from dotmap import DotMap

import clip
from modules.video_clip import video_header
from modules.text_prompt import text_prompt
from Coviar.transforms import get_compress_augmentation, GroupCenterCrop, GroupScale

try:
    from coviar import load as coviar_load
except ImportError:
    from coviar import load as coviar_load


# ─────────────────────────────────────────────────
# 1. Gradient-Based Spatial Attention Map (no hooks needed)
# ─────────────────────────────────────────────────
class GradientAttentionMap:
    """
    Computes spatial attention maps by taking the gradient of the
    classification score w.r.t. the input image pixels, then
    aggregating across channels.
    
    This approach:
      ✅ Works with gradient checkpointing
      ✅ Works in eval mode
      ✅ No hooks needed
      ✅ Produces meaningful spatial maps
    """

    def __init__(self, model, modality='iframe'):
        self.model = model
        self.modality = modality

        if modality == 'iframe':
            self.encoder = model.visual
        elif modality == 'residual':
            self.encoder = model.residual_encoder
        elif modality == 'mv':
            self.encoder = model.mvs_encoder
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _encode(self, x):
        """
        Full forward pass through the appropriate encoder.
        
        Architecture from checkpoint:
          - visual (iframe):       conv1(3ch→768), 12 resblocks, ln_post, proj [768→512]
          - residual_encoder:      conv1(3ch→768), 2 transformer_blocks, ln_post, proj
                                   has conv_2to3 but residual input is already 3ch → SKIP
          - mvs_encoder:           conv_2to3(2ch→3ch), conv1(3ch→768), 2 transformer_blocks, ln_post, proj
        """
        enc = self.encoder

        if self.modality == 'mv':
            # MV: 2ch → conv_2to3 → 3ch → conv1 → transformer
            x = enc.conv_2to3(x)
        elif self.modality == 'residual':
            # Residual: already 3ch → directly to conv1
            pass
        # iframe: already 3ch → directly to conv1 (enc.forward handles it)

        if self.modality == 'iframe':
            # Use the full visual encoder forward
            # Need to handle the 2ch→3ch check inside VisualTransformer.forward
            return enc(x.type(self.model.dtype))
        else:
            # Manual forward for residual_encoder / mvs_encoder
            x = x.type(self.model.dtype)
            x = enc.conv1(x)                             # [T, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)    # [T, width, grid^2]
            x = x.permute(0, 2, 1)                       # [T, grid^2, width]

            # CLS token
            cls_token = enc.class_embedding.to(x.dtype)
            cls_expanded = cls_token + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_expanded, x], dim=1)      # [T, 1+grid^2, width]
            x = x + enc.positional_embedding.to(x.dtype)
            x = enc.ln_pre(x)

            # Transformer blocks: expects (L, N, D)
            x = x.permute(1, 0, 2)                       # [L, T, width]
            for blk in enc.transformer_blocks:
                x = blk(x)
            x = x.permute(1, 0, 2)                       # [T, L, width]

            # CLS output
            x = enc.ln_post(x[:, 0, :])                  # [T, width]
            if enc.proj is not None:
                x = x @ enc.proj                          # [T, embed_dim]
            return x

    def generate(self, input_tensor, target_index=None, text_features=None):
        """
        Compute spatial attention map via input gradients.
        
        Args:
            input_tensor: [T, C, H, W] - raw input frames
            target_index: int - target class index
            text_features: [n_class, D] - text CLS features
            
        Returns:
            cam: numpy [grid_h, grid_w] normalized to [0, 1]
        """
        self.model.zero_grad()

        # Make input require grad so we can compute d(score)/d(input)
        x = input_tensor.clone().detach().float().requires_grad_(True)

        # Forward
        visual_output = self._encode(x)  # [T, D]

        if text_features is not None:
            v_norm = F.normalize(visual_output.float(), dim=-1)
            t_norm = F.normalize(text_features.float(), dim=-1)
            logits = v_norm @ t_norm.T  # [T, n_class]

            if target_index is None:
                target_index = logits.mean(dim=0).argmax().item()

            score = logits[:, target_index].sum()
        else:
            score = visual_output.float().norm(dim=-1).sum()

        score.backward()

        # x.grad shape: [T, C, H, W]
        grad = x.grad.detach().float()  # [T, C, H, W]

        # Aggregate: take absolute value, mean over channels and frames
        # This gives us a spatial map showing which pixels most affect the score
        cam = grad.abs().mean(dim=0).mean(dim=0)  # [H, W]

        # Downsample to patch grid for cleaner visualization
        # ViT-B/16 with 224 input → 14x14 patches
        patch_size = 16
        grid_size = cam.shape[0] // patch_size
        if grid_size > 0:
            cam = cam.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            cam = cam.contiguous().view(grid_size, grid_size, -1).mean(dim=-1)

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def generate_smooth(self, input_tensor, target_index=None, text_features=None,
                        n_samples=10, noise_std=0.1):
        """
        SmoothGrad: average gradients over noisy copies for cleaner maps.
        
        This produces much better visualizations than single-pass gradients.
        """
        device = input_tensor.device
        all_cams = []

        for i in range(n_samples):
            if i == 0:
                noisy_input = input_tensor
            else:
                noise = torch.randn_like(input_tensor) * noise_std
                noisy_input = input_tensor + noise

            cam = self.generate(noisy_input, target_index, text_features)
            all_cams.append(cam)

        avg_cam = np.mean(all_cams, axis=0)

        # Re-normalize
        c_min, c_max = avg_cam.min(), avg_cam.max()
        if c_max > c_min:
            avg_cam = (avg_cam - c_min) / (c_max - c_min)

        return avg_cam


# ─────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def load_single_video_frames(video_path, num_segments, GOP_SIZE=12, accumulate=True):
    iframes, mvs, residuals = [], [], []

    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=30
        )
        total_frames = int(result.stdout.strip())
    except Exception:
        total_frames = num_segments * GOP_SIZE * 2

    seg_size = float(total_frames - 1) / num_segments
    sampled_indices = []
    for seg in range(num_segments):
        frame_idx = int(np.round(seg_size * seg + seg_size / 2))
        frame_idx = min(frame_idx, total_frames - 1)
        frame_idx = max(frame_idx, 1)
        sampled_indices.append(frame_idx)

    for frame_idx in sampled_indices:
        gop_index = frame_idx // GOP_SIZE
        gop_pos = frame_idx % GOP_SIZE
        gop_pos_mv_res = max(gop_pos, 1)

        try:
            iframe = coviar_load(video_path, gop_index, gop_pos, 0, accumulate)
            iframe = iframe[..., ::-1].copy() if iframe is not None else np.zeros((256, 256, 3), dtype=np.uint8)
        except Exception:
            iframe = np.zeros((256, 256, 3), dtype=np.uint8)

        try:
            mv = coviar_load(video_path, gop_index, gop_pos_mv_res, 1, accumulate)
            if mv is not None:
                mv = clip_and_scale(mv, 20)
                mv += 128
                mv = np.minimum(np.maximum(mv, 0), 255).astype(np.uint8)
            else:
                mv = np.zeros((256, 256, 2), dtype=np.uint8)
        except Exception:
            mv = np.zeros((256, 256, 2), dtype=np.uint8)

        try:
            residual = coviar_load(video_path, gop_index, gop_pos_mv_res, 2, accumulate)
            if residual is not None:
                residual += 128
                residual = np.minimum(np.maximum(residual, 0), 255).astype(np.uint8)
            else:
                residual = np.zeros((256, 256, 3), dtype=np.uint8)
        except Exception:
            residual = np.zeros((256, 256, 3), dtype=np.uint8)

        iframes.append(iframe)
        mvs.append(mv)
        residuals.append(residual)

    return iframes, mvs, residuals, sampled_indices


def preprocess_frames(frames, transform, modality='iframe'):
    transformed = transform(frames)
    transformed = np.array(transformed)
    transformed = np.transpose(transformed, (0, 3, 1, 2))
    tensor = torch.from_numpy(transformed).float() / 255.0
    return tensor


# ────────────────────────────────────────────────���
# 3. Visualization
# ─────────────────────────────────────────────────

def apply_colormap_on_image(org_img, activation_map, colormap_name='jet', alpha=0.5):
    H, W = org_img.shape[:2]
    heatmap = cv2.resize(activation_map.astype(np.float32), (W, H),
                         interpolation=cv2.INTER_CUBIC)
    colormap = plt.get_cmap(colormap_name)
    heatmap_colored = colormap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    superimposed = (alpha * heatmap_colored.astype(np.float32) +
                    (1 - alpha) * org_img.astype(np.float32)).astype(np.uint8)
    return superimposed, heatmap


def mv_to_rgb(mv_frame):
    flow = mv_frame.astype(np.float32) - 128.0
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hsv = np.zeros((*mv_frame.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    max_mag = max(mag.max(), 1)
    hsv[..., 2] = (mag / max_mag * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_display_image(orig_frame, modality_name):
    if modality_name == 'mv' and orig_frame.shape[-1] == 2:
        return mv_to_rgb(orig_frame)
    img = orig_frame.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def visualize_single_modality(
    model, attn_map_gen, frames_tensor, original_frames,
    text_features, target_class, modality_name, output_dir,
    num_segments, grid_cols=4, use_smooth=True
):
    device = next(model.parameters()).device
    input_tensor = frames_tensor.to(device)

    per_frame_cams = []
    for t in range(num_segments):
        frame_tensor = input_tensor[t:t+1]
        if use_smooth:
            cam = attn_map_gen.generate_smooth(
                frame_tensor, target_index=target_class,
                text_features=text_features,
                n_samples=15, noise_std=0.15
            )
        else:
            cam = attn_map_gen.generate(
                frame_tensor, target_index=target_class,
                text_features=text_features
            )
        per_frame_cams.append(cam)
        print(f'    Frame {t}: cam range [{cam.min():.4f}, {cam.max():.4f}]')

    mod_dir = os.path.join(output_dir, modality_name)
    os.makedirs(mod_dir, exist_ok=True)

    # Grid visualization
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
        display_img = get_display_image(original_frames[t], modality_name)

        axes[row, col].imshow(display_img)
        axes[row, col].set_title(f'{modality_name} Frame {t}', fontsize=10)
        axes[row, col].axis('off')

        superimposed, heatmap = apply_colormap_on_image(display_img, per_frame_cams[t])
        axes[row + 1, col].imshow(superimposed)
        axes[row + 1, col].set_title(f'Attention Frame {t}', fontsize=10)
        axes[row + 1, col].axis('off')

        cv2.imwrite(os.path.join(mod_dir, f'frame_{t:03d}_original.jpg'),
                     cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(mod_dir, f'frame_{t:03d}_attention.jpg'),
                     cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        heatmap_vis = (plt.get_cmap('jet')(heatmap)[:, :, :3] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(mod_dir, f'frame_{t:03d}_heatmap.jpg'),
                     cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR))

    for t in range(num_segments, grid_rows * grid_cols):
        row = (t // grid_cols) * 2
        col = t % grid_cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')

    plt.suptitle(f'Attention Map: {modality_name} (class: {target_class})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(mod_dir, f'attention_grid_{modality_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Averaged
    avg_cam = np.mean(per_frame_cams, axis=0)
    avg_cam = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(avg_cam, cmap='jet', interpolation='bilinear')
    ax.set_title(f'Averaged Attention: {modality_name}', fontsize=14)
    ax.axis('off')
    plt.savefig(os.path.join(mod_dir, f'attention_averaged_{modality_name}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  [✓] {modality_name} attention maps saved to {mod_dir}')
    return per_frame_cams, avg_cam


def create_combined_visualization(
    all_cams, all_original_frames, modality_names,
    target_class, class_name, output_dir, num_segments
):
    n_mods = len(modality_names)
    show_indices = np.linspace(0, num_segments - 1, min(4, num_segments), dtype=int)
    n_show = len(show_indices)

    fig, axes = plt.subplots(n_mods * 2, n_show,
                             figsize=(n_show * 4, n_mods * 2 * 3.5))
    if n_mods * 2 == 1:
        axes = axes[np.newaxis, :]
    if n_show == 1:
        axes = axes[:, np.newaxis]

    for mod_idx, mod_name in enumerate(modality_names):
        for col_idx, t in enumerate(show_indices):
            display_img = get_display_image(all_original_frames[mod_name][t], mod_name)
            row_orig = mod_idx * 2
            row_cam = mod_idx * 2 + 1

            axes[row_orig, col_idx].imshow(display_img)
            axes[row_orig, col_idx].set_title(f'{mod_name} t={t}', fontsize=9)
            axes[row_orig, col_idx].axis('off')

            superimposed, _ = apply_colormap_on_image(display_img, all_cams[mod_name][t])
            axes[row_cam, col_idx].imshow(superimposed)
            axes[row_cam, col_idx].set_title(f'Attention t={t}', fontsize=9)
            axes[row_cam, col_idx].axis('off')

    plt.suptitle(f'Multi-Modality Attention | Class: {class_name} (idx={target_class})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_combined_all_modalities.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Averaged comparison
    fig, axes = plt.subplots(1, n_mods, figsize=(n_mods * 5, 5))
    if n_mods == 1:
        axes = [axes]
    for mod_idx, mod_name in enumerate(modality_names):
        avg = np.mean(all_cams[mod_name], axis=0)
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
        avg_resized = cv2.resize(avg.astype(np.float32), (224, 224),
                                  interpolation=cv2.INTER_CUBIC)
        axes[mod_idx].imshow(avg_resized, cmap='jet', interpolation='bilinear')
        axes[mod_idx].set_title(f'Avg Attention: {mod_name}', fontsize=12)
        axes[mod_idx].axis('off')

    plt.suptitle(f'Averaged Attention Comparison | Class: {class_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_averaged_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[✓] Combined visualization saved to {output_dir}')


# ─────────────────────────────────────────────────
# 4. Model Loading
# ─────────────────────────────────────────────────

def load_model_and_config(config_path, checkpoint_path, device='cuda'):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    residual_layers = config.network.get('residual_layers_to_use', None)
    mvs_layers = config.network.get('mvs_layers_to_use', None)

    model, clip_state_dict = clip.load(
        config.network.arch, device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st=config.network.joint_st,
        residual_layers_to_use=residual_layers,
        mvs_layers_to_use=mvs_layers,
    )

    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
    )

    model = model.float()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f'Loading checkpoint: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        model_state = ckpt.get('model_state_dict', ckpt)
        new_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f'  Missing keys: {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')

        fusion_state = ckpt.get('fusion_model_state_dict', None)
        if fusion_state:
            new_fusion = {k.replace('module.', ''): v for k, v in fusion_state.items()}
            video_head.load_state_dict(new_fusion, strict=False)

        print(f'  Checkpoint loaded.')
        del ckpt
    else:
        print(f'  [WARNING] No checkpoint at {checkpoint_path}')

    model = model.to(device)
    video_head = video_head.to(device)
    model.eval()
    video_head.eval()

    # Print encoder info for verification
    print(f'  visual encoder: {len(model.visual.transformer.resblocks)} layers')
    if hasattr(model, 'residual_encoder') and model.residual_encoder is not None:
        print(f'  residual_encoder: {len(model.residual_encoder.transformer_blocks)} layers, '
              f'has conv_2to3: {hasattr(model.residual_encoder, "conv_2to3")}')
    if hasattr(model, 'mvs_encoder') and model.mvs_encoder is not None:
        print(f'  mvs_encoder: {len(model.mvs_encoder.transformer_blocks)} layers, '
              f'has conv_2to3: {hasattr(model.mvs_encoder, "conv_2to3")}')

    return model, video_head, config


def get_text_features(model, config, device='cuda'):
    label_list_path = config.data.label_list
    classnames = []

    if label_list_path.endswith('.csv'):
        import csv
        with open(label_list_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    classnames.append(row[1].strip())
                elif len(row) == 1:
                    classnames.append(row[0].strip())
    else:
        with open(label_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    classnames.append(line.split(',')[-1].strip() if ',' in line else line)

    attribute_prompt_cfg = config.network.get('action_prompt',
                                              config.network.get('attribute_prompt', None))
    template = "a video about {}."
    if isinstance(attribute_prompt_cfg, dict):
        template = attribute_prompt_cfg.get('template', template)

    prompts = [template.format(name) for name in classnames]
    text_tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        cls_feature, text_features = model.encode_text(text_tokens, return_token=True)

    return cls_feature, text_features, classnames


# ─────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Gradient-based Attention Visualization')
    parser.add_argument('--config', '-cfg', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--target_class', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='gradcam_output')
    parser.add_argument('--modality', type=str, default='all',
                        choices=['iframe', 'residual', 'mv', 'all'])
    parser.add_argument('--num_segments', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_accumulation', action='store_true')
    parser.add_argument('--grid_cols', type=int, default=4)
    parser.add_argument('--no_smooth', action='store_true',
                        help='Disable SmoothGrad (faster but noisier)')
    parser.add_argument('--smooth_samples', type=int, default=15,
                        help='Number of SmoothGrad samples')
    parser.add_argument('--smooth_noise', type=float, default=0.15,
                        help='SmoothGrad noise standard deviation')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print('=' * 60)
    print('Loading model...')
    model, video_head, config = load_model_and_config(
        args.config, args.checkpoint, device
    )

    num_segments = args.num_segments or config.data.num_segments
    GOP_SIZE = config.data.get('GOP_SIZE', 12)

    has_res = hasattr(model, 'residual_encoder') and model.residual_encoder is not None
    has_mvs = hasattr(model, 'mvs_encoder') and model.mvs_encoder is not None

    print('Computing text features...')
    cls_feature, text_features, classnames = get_text_features(model, config, device)
    n_class = len(classnames)
    print(f'  Classes: {n_class}')

    print(f'Loading video: {args.video_path}')
    accumulate = not args.no_accumulation
    iframes, mvs, residuals, sampled_indices = load_single_video_frames(
        args.video_path, num_segments, GOP_SIZE, accumulate
    )
    print(f'  Loaded {len(iframes)} frames')

    transform_val = get_compress_augmentation(False, config)
    iframe_tensor = preprocess_frames(iframes, transform_val, 'iframe')
    residual_tensor = preprocess_frames(residuals, transform_val, 'residual')
    mv_tensor = preprocess_frames(mvs, transform_val, 'mv')

    print(f'  iframe: {iframe_tensor.shape}, residual: {residual_tensor.shape}, mv: {mv_tensor.shape}')

    # Determine target class
    if args.target_class is not None:
        target_class = args.target_class
    else:
        with torch.no_grad():
            feat = model.visual(iframe_tensor.to(device).type(model.dtype))
            feat_norm = F.normalize(feat.mean(0, keepdim=True), dim=-1)
            cls_norm = F.normalize(cls_feature, dim=-1)
            target_class = (feat_norm @ cls_norm.T).argmax(-1).item()
        print(f'  Predicted class: {target_class} ({classnames[target_class]})')

    class_name = classnames[target_class] if target_class < n_class else f'class_{target_class}'
    print(f'  Target: {target_class} ({class_name})')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.modality == 'all':
        modalities = ['iframe']
        if has_res: modalities.append('residual')
        if has_mvs: modalities.append('mv')
    else:
        modalities = [args.modality]

    tensors = {'iframe': iframe_tensor, 'residual': residual_tensor, 'mv': mv_tensor}
    originals = {'iframe': iframes, 'residual': residuals, 'mv': mvs}

    all_cams = {}
    all_originals = {}

    for mod_name in modalities:
        print(f'\n{"="*40}')
        print(f'Processing modality: {mod_name}')
        print(f'{"="*40}')

        attn_gen = GradientAttentionMap(model, modality=mod_name)

        per_frame_cams, avg_cam = visualize_single_modality(
            model=model,
            attn_map_gen=attn_gen,
            frames_tensor=tensors[mod_name],
            original_frames=originals[mod_name],
            text_features=cls_feature,
            target_class=target_class,
            modality_name=mod_name,
            output_dir=args.output_dir,
            num_segments=num_segments,
            grid_cols=args.grid_cols,
            use_smooth=not args.no_smooth,
        )

        all_cams[mod_name] = per_frame_cams
        all_originals[mod_name] = originals[mod_name]

    if len(modalities) > 1:
        print('\nCreating combined visualization...')
        create_combined_visualization(
            all_cams, all_originals, modalities,
            target_class, class_name, args.output_dir, num_segments
        )

    meta = {
        'video_path': args.video_path,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'target_class': target_class,
        'class_name': class_name,
        'num_segments': num_segments,
        'modalities': modalities,
        'arch': config.network.arch,
        'method': 'SmoothGrad' if not args.no_smooth else 'InputGradient',
        'smooth_samples': args.smooth_samples if not args.no_smooth else 0,
        'sampled_indices': sampled_indices,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Done! Results in: {args.output_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

