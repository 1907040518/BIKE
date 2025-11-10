import json
import re
from typing import Set, Dict

def extract_all_word_forms(word: str) -> Set[str]:
    """提取一个词的所有可能形式"""
    forms = {word.lower()}
    
    # 添加常见变形
    if word.endswith('ing'):
        base = word[:-3]
        forms.add(base)
        forms.add(base + 'e')  # make -> making
        # 双写辅音字母的情况
        if len(base) >= 2 and base[-1] == base[-2]:
            forms.add(base[:-1])  # running -> run
    
    if word.endswith('ed'):
        base = word[:-2]
        forms.add(base)
        forms.add(base + 'e')
    
    if word.endswith('s') and not word.endswith('ss'):
        forms.add(word[:-1])
    
    if word.endswith('es'):
        forms.add(word[:-2])
    
    return forms

def get_class_keywords(class_name: str) -> Set[str]:
    """从类别名中提取所有关键词及其变形"""
    stop_words = {
        'a', 'an', 'the', 'with', 'without', 'from', 'to', 'in', 'on', 
        'at', 'by', 'for', 'of', 'as', 'into', 'onto', 'until', 'so', 
        'but', 'or', 'and', 'that', 'it', 'its', 'is', 'are', 'was',
        'your', 'you', 'because', 'if', 'then', 'does', "doesn't",
        'not', 'enough', 'almost', 'just', 'right', 'out', 'off',
        'up', 'down', 'over', 'under', 'actually', 'can', "can't",
        'number', 'part', 'many', 'one', 'two', 'some', 'any', 'both',
        'while', 'where', 'what', 'how', 'when'
    }
    
    # 提取所有单词
    words = re.findall(r'\b\w+\b', class_name.lower())
    
    # 收集所有关键词及其变形
    all_forms = set()
    for word in words:
        if word not in stop_words and len(word) > 2:
            all_forms.update(extract_all_word_forms(word))
    
    return all_forms

def clean_attributes_strict(data: Dict[str, str]) -> Dict[str, str]:
    """严格清理属性中的重复词汇"""
    cleaned_data = {}
    stats = []
    
    for class_name, attributes in data.items():
        # 获取类别关键词
        keywords = get_class_keywords(class_name)
        
        # 分割属性
        attr_words = attributes.split()
        
        # 过滤重复词汇
        filtered_words = []
        removed_words = []
        
        for word in attr_words:
            word_lower = word.lower()
            word_forms = extract_all_word_forms(word_lower)
            
            # 检查是否与任何关键词的任何形式重叠
            if word_forms & keywords:  # 集合交集
                removed_words.append(word)
            else:
                filtered_words.append(word)
        
        cleaned_attributes = ' '.join(filtered_words)
        cleaned_data[class_name] = cleaned_attributes
        
        # 记录统计
        stats.append({
            'class': class_name,
            'keywords': keywords,
            'original_count': len(attr_words),
            'cleaned_count': len(filtered_words),
            'removed_words': removed_words
        })
    
    return cleaned_data, stats

# 主程序
if __name__ == "__main__":
    # 读取您的数据
    # 方式1: 直接使用变量
    # data = {
    #     "Approaching something with your camera": "approaching camera object interacting manipulating observing capturing tracking analyzing monitoring recording visualizing documenting surveying scanning inspecting exploring mapping detecting identifying processing rendering",
    #     # ... 添加所有您的数据
    # }
    
    # 方式2: 从文件读取
    with open('/home/stu_b/BIKE/attributes/qwen/SOMETHING_SOMETHING_V2_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 清理数据
    cleaned_data, stats = clean_attributes_strict(data)
    
    # 保存结果
    with open('cleaned_attributes.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    # 打印详细报告
    print("=" * 80)
    print("清理报告")
    print("=" * 80)
    
    for stat in stats:
        print(f"\n类别: {stat['class']}")
        print(f"关键词: {stat['keywords']}")
        print(f"原始词数: {stat['original_count']}")
        print(f"清理后词数: {stat['cleaned_count']}")
        print(f"删除词数: {stat['original_count'] - stat['cleaned_count']}")
        if stat['removed_words']:
            print(f"删除的词: {', '.join(stat['removed_words'][:10])}" + 
                  ("..." if len(stat['removed_words']) > 10 else ""))
    
    # 总体统计
    total_original = sum(s['original_count'] for s in stats)
    total_cleaned = sum(s['cleaned_count'] for s in stats)
    print("\n" + "=" * 80)
    print(f"总计: 原始 {total_original} 词 -> 清理后 {total_cleaned} 词")
    print(f"删除比例: {(total_original - total_cleaned) / total_original * 100:.2f}%")
    print("=" * 80)
