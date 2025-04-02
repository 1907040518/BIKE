#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi
export CUDA_VISIBLE_DEVICES=2,3
now=$(date +"%Y%m%d_%H%M%S")
python -m torch.distributed.launch --master_port 1241 --nproc_per_node=2 \
         train_compress_0.4.py  --config ${config} --log_time $now
         