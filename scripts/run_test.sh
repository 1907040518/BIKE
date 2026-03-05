

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=12023 \
  test_comp_CoAPT.py \
  --config /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/20260305_010951/hmdb_CLIP_fix_B16_0_11_fusion.yaml \
  --weights /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/20260305_010951/model_best.pt \
  --test-list /home/neimedia/gmk/BIKE//lists/hmdb51/val_rgb_split_1.txt \
  --save_predictions /home/neimedia/gmk/BIKE/exps/hmdb51/ViT-B/16/20260305_010951/my_1_predictions.txt

