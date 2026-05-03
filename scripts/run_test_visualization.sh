CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=12023 \
  /home/neimedia/gmk/BIKE/visualization_comparison/test_visualization.py \
  --config /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/Video-Text交互/hmdb_CLIP_fix_B16_0_11_fusion.yaml \
  --weights /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/Video-Text交互/model_best.pt \
  --visualize_class sit \
  --visualize_savedir "/home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/Video-Text交互/" \
  --save_predictions /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/Video-Text交互/predictions.txt \
