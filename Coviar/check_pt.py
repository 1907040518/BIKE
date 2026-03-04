import torch
import sys
import os
from contextlib import redirect_stdout

ckpt = torch.load(
    '/home/neimedia/gmk/BIKE/Coviar/hmdb51_iframe_model_iframe_model_best.pth.tar',
    map_location='cpu',
    weights_only=False
)

output_txt = '/home/neimedia/gmk/BIKE/Coviar/hmdb51_iframe_model_info.txt'

def tensor_stats(t):
    """安全地打印 Tensor 统计信息，兼容所有 dtype"""
    if t.numel() == 0:
        return "空 Tensor"
    if t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16,
                   torch.complex64, torch.complex128):
        return (f"Shape: {list(t.shape)}, Dtype: {t.dtype} | "
                f"Min: {t.min().item():.6f}, Max: {t.max().item():.6f}, Mean: {t.mean().item():.6f}")
    elif t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8,
                     torch.uint8, torch.bool):
        if t.numel() == 1:
            return f"Shape: {list(t.shape)}, Dtype: {t.dtype} | Value: {t.item()}"
        return (f"Shape: {list(t.shape)}, Dtype: {t.dtype} | "
                f"Min: {t.min().item()}, Max: {t.max().item()}, Sum: {t.sum().item()}")
    else:
        return f"Shape: {list(t.shape)}, Dtype: {t.dtype}"

def print_all(f):
    # 顶层键
    print("=" * 60, file=f)
    print("顶层键 (Top-level keys):", file=f)
    print("=" * 60, file=f)
    for key in ckpt.keys():
        print(f"  {key}  ->  Type: {type(ckpt[key]).__name__}", file=f)

    print(file=f)

    # 详细内容
    print("=" * 60, file=f)
    print("详细内容:", file=f)
    print("=" * 60, file=f)
    for key, val in ckpt.items():
        if isinstance(val, torch.Tensor):
            print(f"[Tensor] {key}", file=f)
            print(f"         {tensor_stats(val)}", file=f)

        elif isinstance(val, dict):
            print(f"\n[Dict]   {key}  ->  包含 {len(val)} 个子键", file=f)
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, torch.Tensor):
                    print(f"  ├─ {sub_key}", file=f)
                    print(f"     {tensor_stats(sub_val)}", file=f)
                else:
                    print(f"  ├─ {sub_key}  ->  {type(sub_val).__name__}: {sub_val}", file=f)

        elif isinstance(val, (int, float, str, bool)):
            print(f"[Scalar] {key}  ->  {val}", file=f)

        elif isinstance(val, list):
            print(f"[List]   {key}  ->  长度: {len(val)}, 内容: {val}", file=f)

        else:
            print(f"[Other]  {key}  ->  Type: {type(val).__name__}, Value: {val}", file=f)

    print(file=f)
    print("=" * 60, file=f)
    print("完成！", file=f)


# 同时输出到终端 + 保存到文件
class Tee:
    """同时写入终端和文件"""
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

with open(output_txt, 'w', encoding='utf-8') as f:
    tee = Tee(f)
    sys.stdout = tee
    print_all(f)
    sys.stdout = tee.terminal  # 恢复标准输出

print(f"\n✅ 结果已保存至: {output_txt}")
