#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
export CUDA_VISIBLE_DEVICES=4,5
# 使用torchrun替代torch.distributed.launch
torchrun --nproc_per_node=2 --master_port=12045 \
         train_comp_CoAPT.py --config ${config} --log_time $now
