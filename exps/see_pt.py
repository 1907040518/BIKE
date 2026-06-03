import os

try:
    import torch
except ImportError:
    print("错误: 未找到 torch 库。请使用 'pip install torch' 安装。")
    exit(1)


def format_tensor_info(tensor, mode="stats", preview_count=8):
    """
    根据指定模式格式化张量信息。

    Args:
        tensor (torch.Tensor): 要格式化的张量。
        mode (str): 输出模式。
            - "stats"   : 仅输出统计信息（均值、标准差、最大值、最小值）
            - "preview" : 输出前 preview_count 个扁平化数值
            - "full"    : 输出所有数值（谨慎使用！）
        preview_count (int): "preview" 模式下显示的数值数量。

    Returns:
        str: 格式化后的张量信息字符串。
    """
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype)
    numel = tensor.numel()

    # 转为 float32 以便统计计算（避免 bfloat16 等精度问题）
    t = tensor.float()

    lines = []
    lines.append(f"  Shape : {shape_str}")
    lines.append(f"  Dtype : {dtype_str}")
    lines.append(f"  Numel : {numel:,} 个元素")

    if mode == "stats" or mode == "preview" or mode == "full":
        # 统计信息始终输出
        lines.append(f"  Mean  : {t.mean().item():.6f}")
        lines.append(f"  Std   : {t.std().item():.6f}")
        lines.append(f"  Min   : {t.min().item():.6f}")
        lines.append(f"  Max   : {t.max().item():.6f}")

    if mode == "preview":
        flat = t.flatten()
        count = min(preview_count, numel)
        vals = [f"{v:.6f}" for v in flat[:count].tolist()]
        lines.append(f"  前{count}个值: [{', '.join(vals)}{'...' if numel > count else ''}]")

    elif mode == "full":
        flat = t.flatten().tolist()
        # 每行输出8个数值，保持可读性
        chunk_size = 8
        lines.append(f"  全部数值 ({numel:,} 个):")
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i + chunk_size]
            vals_str = "  ".join(f"{v:12.6f}" for v in chunk)
            lines.append(f"    [{i:>8}] {vals_str}")

    return "\n".join(lines)


def get_all_keys_with_shapes(data, prefix="", mode="stats", preview_count=8):
    """
    递归地遍历字典，提取所有键及其对应的形状和权重值信息。

    Args:
        data: 当前正在检查的项（字典或张量等）。
        prefix: 嵌套键名的累积前缀。
        mode (str): 张量输出模式（"stats" / "preview" / "full"）。
        preview_count (int): preview 模式下显示的数值数量。

    Returns:
        list: 包含格式化字符串的列表。
    """
    all_keys_shapes = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                all_keys_shapes.extend(
                    get_all_keys_with_shapes(value, prefix=current_path,
                                             mode=mode, preview_count=preview_count)
                )
            elif isinstance(value, torch.Tensor):
                tensor_info = format_tensor_info(value, mode=mode,
                                                 preview_count=preview_count)
                all_keys_shapes.append(
                    f"【{current_path}】\n{tensor_info}"
                )
            else:
                all_keys_shapes.append(
                    f"【{current_path}】\n  Type  : {type(value).__name__}\n  Value : {repr(value)}"
                )

    return all_keys_shapes


def inspect_and_save_pt_shapes(pt_file_path, output_txt_path,
                                mode="stats", preview_count=8):
    """
    加载 .pt 文件，递归提取所有键、形状和权重信息，并保存到文本文件。

    Args:
        pt_file_path (str): .pt 文件的路径。
        output_txt_path (str): 要保存结果的输出 .txt 文件路径。
        mode (str): 张量输出模式。
            - "stats"   : 统计信息（均值/标准差/最大/最小）【推荐，文件小】
            - "preview" : 统计信息 + 前N个数值预览
            - "full"    : 统计信息 + 全部数值【文件可能极大，谨慎！】
        preview_count (int): "preview" 模式下显示的数值数量。
    """
    if not os.path.exists(pt_file_path):
        print(f"错误: 找不到文件 {pt_file_path}")
        return

    print(f"正在加载 {pt_file_path} ... 这可能需要一点时间。")
    print(f"输出模式: [{mode}]")

    try:
        data = torch.load(pt_file_path, weights_only=False, map_location='cpu')

        keys_and_shapes = get_all_keys_with_shapes(
            data, mode=mode, preview_count=preview_count
        )

        if not keys_and_shapes:
            print(f"无法提取信息。加载的 .pt 文件可能包含类型为 {type(data)} 的对象，而不是状态字典。")
            return

        print(f"成功提取了 {len(keys_and_shapes)} 个参数的信息。")

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"文件: {os.path.basename(pt_file_path)}\n")
            f.write(f"输出模式: {mode}\n")
            f.write(f"参数总数: {len(keys_and_shapes)}\n")
            f.write("=" * 80 + "\n\n")
            for item in keys_and_shapes:
                f.write(f"{item}\n")
                f.write("-" * 60 + "\n")

        print(f"\n✅ 所有参数信息已成功保存到: {output_txt_path}")

    except Exception as e:
        print(f"加载或处理 .pt 文件时发生错误: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    INPUT_PT_FILE = "/home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/final_IADF/model_best.pt"
    OUTPUT_TEXT_FILE = "/home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/final_IADF/model_best_weights.txt"

    # ✅ 模式选择（三选一）：
    # "stats"   → 只输出统计量，文件约几百KB，速度最快  【推荐日常使用】
    # "preview" → 统计量 + 前8个数值，文件约几MB
    # "full"    → 全部数值，文件可能超过 1GB，慎用！
    MODE = "preview"

    # preview 模式下每个参数显示多少个数值
    PREVIEW_COUNT = 8

    inspect_and_save_pt_shapes(INPUT_PT_FILE, OUTPUT_TEXT_FILE,
                                mode=MODE, preview_count=PREVIEW_COUNT)
