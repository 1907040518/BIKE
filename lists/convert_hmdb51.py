#!/usr/bin/env python3
"""
HMDB51数据格式转换脚本   把生成的字幕json文件转为BIKE可以识别的文件
将hmdb51_mul_test.json中的sentence信息替换到test_caption.json的semantic_description字段

使用方法:
python convert_hmdb51.py
"""

import json
import os

def convert_hmdb51_format():
    """
    将hmdb51_mul_test.json中的sentence信息替换到test_caption.json中
    """
    
    # 文件路径
    hmdb51_file = "caption_train.json"
    test_caption_file = "train_caption.json"
    output_file = "mul_caption_train.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(hmdb51_file):
        raise FileNotFoundError(f"文件不存在: {hmdb51_file}")
    if not os.path.exists(test_caption_file):
        raise FileNotFoundError(f"文件不存在: {test_caption_file}")
    
    # 读取两个JSON文件
    print(f"读取源数据文件: {hmdb51_file}")
    with open(hmdb51_file, 'r', encoding='utf-8') as f:
        hmdb51_data = json.load(f)
    
    print(f"读取目标格式文件: {test_caption_file}")
    with open(test_caption_file, 'r', encoding='utf-8') as f:
        test_caption_data = json.load(f)
    
    # 创建一个从文件路径到sentence的映射字典
    path_to_sentence = {}
    
    for file_path, content_list in hmdb51_data["results"].items():
        # 获取sentence（使用第一个条目的sentence字段）
        if content_list and len(content_list) > 0:
            sentence = content_list[0]["sentence"]
            path_to_sentence[file_path] = sentence
    
    print(f"从hmdb51文件中提取到 {len(path_to_sentence)} 个路径映射")
    
    # 复制test_caption_data并替换semantic_description
    result_data = {}
    matched_count = 0
    unmatched_files = []
    
    for file_path, content in test_caption_data.items():
        # 复制原始内容
        new_content = content.copy()
        
        # 查找对应的sentence
        if file_path in path_to_sentence:
            new_content["semantic_description"] = path_to_sentence[file_path]
            matched_count += 1
        else:
            # 如果没找到匹配，保留原始描述
            unmatched_files.append(file_path)
        
        result_data[file_path] = new_content
    
    # 保存结果
    print(f"保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 输出统计信息
    print(f"\n=== 转换完成 ===")
    print(f"总文件数: {len(test_caption_data)}")
    print(f"成功匹配: {matched_count}")
    print(f"未匹配: {len(unmatched_files)}")
    print(f"匹配率: {matched_count/len(test_caption_data)*100:.1f}%")
    
    # 显示一些转换示例
    print(f"\n=== 转换示例 ===")
    example_count = 0
    for file_path, content in result_data.items():
        if file_path in path_to_sentence:
            original_desc = test_caption_data[file_path]["semantic_description"]
            new_desc = content["semantic_description"]
            print(f"文件: {file_path}")
            print(f"  原始描述: {original_desc}")
            print(f"  新描述: {new_desc}")
            print()
            example_count += 1
            if example_count >= 3:  # 只显示前3个示例
                break
    
    if unmatched_files:
        print(f"\n未匹配的文件 (前5个):")
        for file in unmatched_files[:5]:
            print(f"  {file}")
        if len(unmatched_files) > 5:
            print(f"  ... 还有 {len(unmatched_files) - 5} 个")
    
    return result_data

def main():
    try:
        convert_hmdb51_format()
        print(f"\n✅ 转换成功! 结果已保存到: caption_test.json")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
