import os

try:
    import torch
except ImportError:
    print("错误: 未找到 torch 库。请使用 'pip install torch' 安装。")
    exit(1)


def get_all_keys_with_shapes(data, prefix=""):
    """
    递归地遍历字典，提取所有键及其对应的形状（如果是张量）。
    如果是嵌套字典，键名将被扁平化。

    Args:
        data: 当前正在检查的项（字典或张量等）。
        prefix: 嵌套键名的累积前缀。

    Returns:
        list: 包含格式化字符串（键名: 形状）的列表。
    """
    all_keys_shapes = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else str(key)
            
            # 如果是字典，递归深入
            if isinstance(value, dict):
                all_keys_shapes.extend(get_all_keys_with_shapes(value, prefix=current_path))
            # 如果是张量，记录其形状
            elif isinstance(value, torch.Tensor):
                shape_str = str(list(value.shape))
                all_keys_shapes.append(f"{current_path}  ->  Shape: {shape_str}")
            # 如果是其他类型，仅记录键和类型
            else:
                 all_keys_shapes.append(f"{current_path}  ->  Type: {type(value).__name__}")
                 
    return all_keys_shapes

def inspect_and_save_pt_shapes(pt_file_path, output_txt_path):
    """
    加载 .pt 文件，递归提取所有键和形状，并保存到文本文件。

    Args:
        pt_file_path (str): .pt 文件的路径。
        output_txt_path (str): 要保存结果的输出 .txt 文件路径。
    """
    if not os.path.exists(pt_file_path):
        print(f"错误: 找不到文件 {pt_file_path}")
        return

    print(f"正在加载 {pt_file_path} ...这可能需要一点时间。")
    
    try:
        data = torch.load(pt_file_path, weights_only=False, map_location='cpu')
        
        # 提取所有键和形状
        keys_and_shapes = get_all_keys_with_shapes(data)
        
        if not keys_and_shapes:
             print(f"无法提取信息。加载的 .pt 文件可能包含类型为 {type(data)} 的对象，而不是状态字典。")
             return

        print(f"成功提取了 {len(keys_and_shapes)} 个参数的形状信息。")
        
        # 保存所有信息
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"在文件 {os.path.basename(pt_file_path)} 中找到的参数及形状:\n")
            f.write("="*80 + "\n")
            for item in keys_and_shapes:
                f.write(f"{item}\n")
                
        print(f"\n所有参数形状已成功保存到: {output_txt_path}")

    except Exception as e:
        print(f"加载或处理 .pt 文件时发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    INPUT_PT_FILE = "/home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/20260509_211709/model_best.pt"  
    OUTPUT_TEXT_FILE = "/home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/20260509_211709/model_best_shapes_list.txt" 
    
    inspect_and_save_pt_shapes(INPUT_PT_FILE, OUTPUT_TEXT_FILE)