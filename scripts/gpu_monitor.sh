#!/bin/bash
# nohup bash scripts/gpu_monitor.sh >> ./exps/gpu.log 2>&1 & 

training_done=0  # 初始化状态标志

while true
do
  # shellcheck disable=SC1060
  stat_arr=()
  
  # 提取每张 GPU 的显存占用，并输出
  for i in {1..8}; do 
    stat=$(gpustat | awk "{print \$09}" | sed -n "${i}p")
    stat_arr+=($stat)  # 将显存占用添加到数组
    echo "GPU $((i-1)) 显存占用: $stat MB"  # 输出每张 GPU 的显存占用
  done

  gpu_available=0
  gpu_available_index_arr=()

  for i in ${!stat_arr[@]}
  do
    if [ "${stat_arr[$i]}" -lt 1 ]; then
      gpu_available=$((gpu_available + 1))
      gpu_available_index_arr[${#gpu_available_index_arr[@]}]=$i
    fi
  done
  
  echo '-可用GPU数:'$gpu_available', 第'${gpu_available_index_arr[@]}'块GPU可用'
  
  if [ $gpu_available -ge 7 ]; then
    if [ $training_done -eq 0 ]; then
      echo 'start running the code number 1 ...'
      # 替换你自己的命令
      # nohup sh scripts/run_train_compress.sh configs/hmdb51/hmdb_k400_finetune_compress.yaml >> ./exps/hmdb51/ViT-L/14/7_pretrain_iframe_cls_video.log 2>&1 & 
      # nohup sh scripts/run_train_compress.sh configs/hmdb51/hmdb_k400_finetune_compress.yaml >> ./exps/hmdb51/ViT-L/14/7_iframe_CROSS6layer.log 2>&1 
      nohup sh scripts/run_train_compress.sh configs/hmdb51/hmdb_k400_finetune.yaml >> ./exps/hmdb51/ViT-L/14/7_rgb.log 2>&1 &
      training_done=1  # 设置状态标志为已执行
      # exit 0
    elif [ $training_done -eq 1 ]; then
      echo 'start running the code number 2 ...'
    #   # 替换你自己的命令
    #   nohup sh scripts/run_train_compress.sh configs/hmdb51/hmdb_k400_finetune_compress_GOP.yaml >> ./exps/hmdb51/ViT-L/14/7_pretrain_iframe_GOP16.log 2>&1 & 
      training_done=2  # 设置状态标志为调试命令已执行
    fi
  fi

  # 在执行完调试命令后退出脚本
  if [ $training_done -eq 2 ]; then
    echo '所有命令已执行完成，脚本将退出。'
    exit 0
  fi
  
  sleep 300
done
