

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12023 \
  test_comp_CoAPT.py \
  --config /home/stu_b/BIKE/configs/hmdb51/hmdb_CLIP_fix_B16_0_11_new.yaml \
  --weights /home/stu_b/BIKE/exps/hmdb51/ViT-B/16/20251108_161046/model_best.pt \
  --test-list /home/stu_b/BIKE/lists/hmdb51/val_rgb_split_2.txt

