

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=12023 \
  test_comp_CoAPT.py \
  --config /home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/20260507_001605/sthv2_pre_fix_B16.yaml \
  --weights /home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/20260507_001605/model_best.pt \
  --test-list /home/neimedia/gmk/BIKE/lists/sthv2/val_rgb.txt \
  --save_predictions /home/neimedia/gmk/BIKE/exps/sthv2/ViT-B/16/20260507_001605/my_1_predictions.txt

