#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES=5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # ← 必须 export

torchrun --nproc_per_node=3 --master_port=12567 \
         train_comp_CoAPT.py --config ${config} --log_time $now
