import json
import csv

# 1. 读取labels.csv文件，建立id到name的映射
label_dict = {}
with open('/home/stu_b/BIKE/lists/k400/kinetics_400_labels.csv', 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # 跳过表头
    for row in csv_reader:
        label_dict[int(row[0])] = row[1]

# 2. 读取JSON文件
with open('/home/stu_b/BIKE/lists/k400/K400_val_vit_L14_attributes.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. 为每个视频添加label_name字段
for video_id, video_info in data.items():
    label_id = video_info['video_label_id']
    video_info['video_label_name'] = label_dict.get(label_id, 'unknown')

# 4. 保存格式化后的JSON文件
with open('/home/stu_b/BIKE/lists/k400/K400_val_vit_L14_attributes_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("处理完成！已生成 K400_val_vit_L14_attributes_formatted.json")
print(f"共处理 {len(data)} 个视频条目")
