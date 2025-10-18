import json
import re

def parse_qwen_txt(file_path):
    """
    解析 qwen.txt 文件，提取视频路径和描述
    返回字典: {视频路径: 描述}
    """
    video_descriptions = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割每个视频块
    video_blocks = content.split('─' * 80)
    
    for block in video_blocks:
        if not block.strip():
            continue
        
        # 提取视频路径
        video_match = re.search(r'📁 视频 \d+: (.+)', block)
        if not video_match:
            continue
        
        video_path = video_match.group(1).strip()
        
        # 提取 Qwen 视频描述
        desc_match = re.search(r'🤖 Qwen视频描述:\s*\n(.+?)(?=\n─|$)', block, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
            video_descriptions[video_path] = description
    
    return video_descriptions

def update_json_with_descriptions(json_path, video_descriptions, output_path=None):
    """
    更新 JSON 文件中的 semantic_description 字段
    
    参数:
        json_path: 原始 JSON 文件路径
        video_descriptions: 从 qwen.txt 解析出的描述字典
        output_path: 输出文件路径，如果为 None 则覆盖原文件
    """
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计更新情况
    updated_count = 0
    not_found_count = 0
    not_found_videos = []
    
    # 更新描述
    for video_path in data.keys():
        if video_path in video_descriptions:
            data[video_path]['semantic_description'] = video_descriptions[video_path]
            updated_count += 1
        else:
            not_found_count += 1
            not_found_videos.append(video_path)
    
    # 保存更新后的 JSON
    if output_path is None:
        output_path = json_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"✅ 更新完成！")
    print(f"📊 总视频数: {len(data)}")
    print(f"✔️  成功更新: {updated_count}")
    print(f"❌ 未找到匹配: {not_found_count}")
    
    if not_found_videos and not_found_count <= 10:
        print(f"\n未找到匹配的视频:")
        for video in not_found_videos:
            print(f"  - {video}")
    
    return data

def main():
    # 文件路径配置
    qwen_txt_path = '/home/stu_b/BIKE/lists/hmdb51/qwen_video_resultstest.txt'
    json_path = '/home/stu_b/BIKE/lists/hmdb51/test_caption.json'
    output_json_path = '/home/stu_b/BIKE/lists/hmdb51/test_qwen.json'  # 可以改为 None 来覆盖原文件
    
    print("🔄 开始解析 qwen_video_results.txt 文件...")
    video_descriptions = parse_qwen_txt(qwen_txt_path)
    print(f"✅ 成功解析 {len(video_descriptions)} 个视频描述\n")
    
    print("🔄 开始更新 JSON 文件...")
    update_json_with_descriptions(json_path, video_descriptions, output_json_path)
    print(f"\n💾 结果已保存到: {output_json_path}")

if __name__ == "__main__":
    main()
