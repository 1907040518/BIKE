#!/usr/bin/env bash

# ============================================
# 配置参数
# ============================================
if [ -f "$1" ]; then
  config="$1"
else
  echo "错误: 需要提供配置文件路径"
  echo "用法: bash 'scripts/run_attribute_select copy.sh' <config_file> <attributes_file>"
  exit 1
fi

if [ -f "$2" ]; then
  attributes="$2"
else
  echo "错误: 需要提供属性JSON文件路径"
  echo "用法: bash 'scripts/run_attribute_select copy.sh' <config_file> <attributes_file>"
  exit 1
fi

# ============================================
# 训练参数设置
# ============================================
now=$(date +"%Y%m%d_%H%M%S")
output_dir="results/attribute_search_${now}"
mkdir -p "${output_dir}"

# GPU设置（使用4号和5号GPU）
export CUDA_VISIBLE_DEVICES=2,3,4,5
num_gpus=4

# 训练超参数
batch_size=32          # 每个GPU的batch size
num_workers=4         # 每个GPU的数据加载线程数
epochs=20
max_motion=1
max_appearance=1
n_ctx=4
n_att=2
prompt_lr=1e-3
weight_lr=1e-3
weight_entropy=5e-2
combo_chunk_size=8
topk=10

# ============================================
# 启动分布式训练
# ============================================
echo "=========================================="
echo "开始属性搜索训练"
echo "=========================================="
echo "配置文件: ${config}"
echo "属性文件: ${attributes}"
echo "输出目录: ${output_dir}"
echo "可见GPU: ${CUDA_VISIBLE_DEVICES} (总数: ${num_gpus})"
echo "Batch Size (per GPU): ${batch_size}"
echo "=========================================="

torchrun \
    --nproc_per_node=${num_gpus} \
    --master_port=29512 \
    train_attribute_qwen.py \
  --config "${config}" \
  --attributes "${attributes}" \
  --batch-size ${batch_size} \
  --num-workers ${num_workers} \
    --epochs ${epochs} \
    --max-motion ${max_motion} \
    --max-appearance ${max_appearance} \
    --n-ctx ${n_ctx} \
    --n-att ${n_att} \
    --prompt-lr ${prompt_lr} \
    --weight-lr ${weight_lr} \
  --weight-entropy ${weight_entropy} \
    --combo-chunk-size ${combo_chunk_size} \
    --topk ${topk} \
  --output "${output_dir}/results.json" \
  2>&1 | tee "${output_dir}/train.log"

echo "=========================================="
echo "训练完成！"
echo "结果保存在: ${output_dir}"
echo "=========================================="
