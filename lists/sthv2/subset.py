import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def load_sthv2_txt(txt_path):
    """加载txt格式的SST-V2标注文件"""
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                video_id, template, label = parts
                data.append({
                    'id': video_id,
                    'template': int(template),
                    'label': int(label)
                })
    return pd.DataFrame(data)

def create_sthv2_subset(
    input_txt_path,
    output_txt_path,
    sampling_ratio=0.1,
    random_state=42,
    min_samples_per_class=1
):
    """
    创建SST-V2子集
    
    参数:
        input_txt_path: 输入txt文件路径
        output_txt_path: 输出txt文件路径
        sampling_ratio: 抽样比例 (0-1)
        random_state: 随机种子
        min_samples_per_class: 每个类别最少保留的样本数
    """
    
    print("=" * 60)
    print("开始创建 SST-V2 子集")
    print("=" * 60)
    
    # 1. 加载数据
    print(f"\n📂 加载数据: {input_txt_path}")
    df = load_sthv2_txt(input_txt_path)
    
    print(f"✓ 原始数据: {len(df)} 个视频")
    print(f"✓ 类别数量: {df['label'].nunique()} 个")
    
    # 2. 检查类别分布
    label_counts = df['label'].value_counts().sort_index()
    print(f"\n📊 类别分布统计:")
    print(f"  - 最少样本数: {label_counts.min()}")
    print(f"  - 最多样本数: {label_counts.max()}")
    print(f"  - 平均样本数: {label_counts.mean():.1f}")
    
    # 3. 分层抽样
    print(f"\n🎯 开始分层抽样 (比例: {sampling_ratio*100:.1f}%)")
    
    try:
        # 使用sklearn的分层抽样
        df_subset, _ = train_test_split(
            df,
            train_size=sampling_ratio,
            stratify=df['label'],
            random_state=random_state
        )
        
        # 确保每个类别至少有min_samples_per_class个样本
        subset_label_counts = df_subset['label'].value_counts()
        if subset_label_counts.min() < min_samples_per_class:
            print(f"⚠️  某些类别样本过少，调整抽样策略...")
            df_subset = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), max(min_samples_per_class, int(len(x) * sampling_ratio))),
                    random_state=random_state
                )
            ).reset_index(drop=True)
    
    except ValueError as e:
        print(f"⚠️  标准分层抽样失败 (可能某些类别样本太少): {e}")
        print(f"   使用备选方案: 按比例从每个类别抽样...")
        df_subset = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max(min_samples_per_class, int(len(x) * sampling_ratio))),
                random_state=random_state
            )
        ).reset_index(drop=True)
    
    # 4. 保存子集
    print(f"\n💾 保存子集到: {output_txt_path}")
    with open(output_txt_path, 'w') as f:
        for _, row in df_subset.iterrows():
            f.write(f"{row['id']} {row['template']} {row['label']}\n")
    
    # 5. 统计信息
    print(f"\n" + "=" * 60)
    print("✅ 子集创建完成!")
    print("=" * 60)
    print(f"📈 数据统计:")
    print(f"  原始视频数: {len(df)}")
    print(f"  子集视频数: {len(df_subset)}")
    print(f"  实际抽样比例: {len(df_subset)/len(df)*100:.2f}%")
    
    print(f"\n📊 类别统计:")
    print(f"  原始类别数: {df['label'].nunique()}")
    print(f"  子集类别数: {df_subset['label'].nunique()}")
    print(f"  类别覆盖率: {df_subset['label'].nunique()/df['label'].nunique()*100:.1f}%")
    
    # 6. 验证分布一致性
    print(f"\n🔍 分布验证 (前10个类别):")
    print(f"{'类别':<8} {'原始数量':<12} {'子集数量':<12} {'比例':<10}")
    print("-" * 45)
    
    original_dist = df['label'].value_counts().sort_index()
    subset_dist = df_subset['label'].value_counts().sort_index()
    
    for label in sorted(df['label'].unique())[:10]:
        orig_count = original_dist.get(label, 0)
        sub_count = subset_dist.get(label, 0)
        ratio = sub_count / orig_count * 100 if orig_count > 0 else 0
        print(f"{label:<8} {orig_count:<12} {sub_count:<12} {ratio:.1f}%")
    
    # 7. 保存视频ID列表 (可选，用于复制视频文件)
    video_ids_path = output_txt_path.replace('.txt', '_video_ids.txt')
    with open(video_ids_path, 'w') as f:
        for video_id in df_subset['id']:
            f.write(f"{video_id}\n")
    print(f"\n📝 视频ID列表已保存到: {video_ids_path}")
    
    return df_subset

# ============== 主程序 ==============

if __name__ == "__main__":
    
    # 配置参数
    TRAIN_TXT = "/home/stu_b/BIKE-main/lists/sthv2/train_rgb.txt"  # 你的训练集标注文件
    VAL_TXT = "/home/stu_b/BIKE-main/lists/sthv2/val_rgb.txt"      # 你的验证集标注文件 (如果有)
    
    OUTPUT_TRAIN = "/home/stu_b/BIKE-main/lists/sthv2/train_subset_5pct.txt"
    OUTPUT_VAL = "/home/stu_b/BIKE-main/lists/sthv2/val_subset_5pct.txt"
    
    SAMPLING_RATIO = 0.05  # 10%
    RANDOM_SEED = 42
    
    # 创建训练集子集
    print("\n" + "🎬" * 30)
    print("处理训练集")
    print("🎬" * 30)
    # 创建多个不同的子集用于交叉验证
    for seed in [42, 123, 456]:
        create_sthv2_subset(
            input_txt_path=TRAIN_TXT,
            output_txt_path=f"train_subset_seed{seed}.txt",
            sampling_ratio=0.05,
            random_state=seed
        )


    # 创建验证集子集 (如果有验证集)
    try:
        print("\n\n" + "🎬" * 30)
        print("处理验证集")
        print("🎬" * 30)
        for seed in [42, 123, 456]:
            create_sthv2_subset(
                input_txt_path=VAL_TXT,
                output_txt_path=f"val_subset_seed{seed}.txt",
                sampling_ratio=0.05,
                random_state=seed
            )
    except FileNotFoundError:
        print(f"\n⚠️  未找到验证集文件: {VAL_TXT}")
        print("   如果不需要验证集子集，可以忽略此消息")
    
    print("\n\n" + "🎉" * 30)
    print("所有子集创建完成!")
    print("🎉" * 30)
