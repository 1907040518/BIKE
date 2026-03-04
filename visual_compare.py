"""
combine_comparison.py
将 ViT (grad_cam.py) 和 ResNet (grad_cam_resnet.py) 的结果拼成一张对比图
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_comparison_figure(
    vit_dir,      # grad_cam.py 输出的 iframe 子目录
    resnet_dir,   # grad_cam_resnet.py 输出的 iframe 子目录
    num_frames=4,
    output_path='comparison_vit_vs_resnet.png'
):
    fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 4, 12))

    for t in range(num_frames):
        # 原始帧（用 ViT 的，两者相同）
        orig = cv2.imread(
            os.path.join(vit_dir, f'frame_{t:03d}_original.jpg')
        )
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # ViT 注意力图
        vit_attn = cv2.imread(
            os.path.join(vit_dir, f'frame_{t:03d}_attention.jpg')
        )
        vit_attn = cv2.cvtColor(vit_attn, cv2.COLOR_BGR2RGB)

        # ResNet Grad-CAM
        res_cam = cv2.imread(
            os.path.join(resnet_dir, f'frame_{t:03d}_attention.jpg')
        )
        res_cam = cv2.cvtColor(res_cam, cv2.COLOR_BGR2RGB)

        axes[0, t].imshow(orig);      axes[0, t].axis('off')
        axes[1, t].imshow(vit_attn);  axes[1, t].axis('off')
        axes[2, t].imshow(res_cam);   axes[2, t].axis('off')

        if t == 0:
            axes[0, t].set_ylabel('Original',     fontsize=13)
            axes[1, t].set_ylabel('ViT (Ours)',   fontsize=13)
            axes[2, t].set_ylabel('ResNet152',    fontsize=13)

        axes[0, t].set_title(f'Frame {t}', fontsize=11)

    plt.suptitle('Attention Map Comparison: ViT vs ResNet152', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[✓] 对比图已保存: {output_path}")


make_comparison_figure(
    vit_dir    = '/home/neimedia/gmk/BIKE/visualization/iframe',
    resnet_dir = '/home/neimedia/gmk/BIKE/Coviar/gradcam_resnet_output/iframe',
    num_frames = 4,
    output_path= '/home/neimedia/gmk/BIKE/visualization_comparison/vit_vs_resnet_comparison.png'
)
