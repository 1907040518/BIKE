import json
import os

def update_text_class_with_paths(text_class_file, val_file, output_file):
    """
    根据val.txt文件中的路径信息，更新text_class.json中的文件名为完整路径
    
    Args:
        text_class_file: text_class.json文件路径
        val_file: val.txt文件路径  
        output_file: 输出的更新后的json文件路径
    """
    
    # 读取text_class.json
    with open(text_class_file, 'r', encoding='utf-8') as f:
        text_class_data = json.load(f)
    
    # 读取val.txt并建立文件名到完整路径的映射
    filename_to_path = {}
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 1:
                    full_path = parts[0]  # 完整路径，如 brush_hair/Aussie_Brunette_...
                    filename = os.path.basename(full_path)  # 提取文件名
                    filename_to_path[filename] = full_path
    
    # 更新text_class.json
    updated_data = {}
    
    for old_key, value in text_class_data.items():
        # 查找对应的完整路径
        if old_key in filename_to_path:
            new_key = filename_to_path[old_key]
            updated_data[new_key] = value
            print(f"更新: {old_key} -> {new_key}")
        else:
            # 如果在val.txt中找不到对应的路径，保持原样并给出警告
            updated_data[old_key] = value
            print(f"警告: 未找到 {old_key} 的完整路径，保持原样")
    
    # 保存更新后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n更新完成！结果已保存到 {output_file}")
    print(f"原始条目数: {len(text_class_data)}")
    print(f"更新后条目数: {len(updated_data)}")

def main():
    # 文件路径配置
    text_class_file = "/home/stu_b/BIKE/lists/hmdb51/hmdb51_val_descriptions.json"  # 输入的text_class.json文件
    val_file = "/home/stu_b/BIKE/lists/hmdb51/val_rgb_split_1.txt"                 # 输入的val.txt文件
    output_file = "/home/stu_b/BIKE/lists/hmdb51/test_description.json"  # 输出的更新后的json文件
    
    # 检查文件是否存在
    if not os.path.exists(text_class_file):
        print(f"错误: 找不到文件 {text_class_file}")
        return
    
    if not os.path.exists(val_file):
        print(f"错误: 找不到文件 {val_file}")
        return
    
    # 执行更新
    update_text_class_with_paths(text_class_file, val_file, output_file)

if __name__ == "__main__":
    main()
