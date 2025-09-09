python /home/stu_b/BIKE/qwen/ProcessorWrapper.py \
    --image_lmdb /mnt/data/hmdb51/HMDB51_lmdb/hmdb51_compressed_frames.lmdb \
    --res_lmdb /mnt/data/hmdb51/HMDB51_lmdb/hmdb51_compressed_residuals.lmdb \
    --mv_lmdb /mnt/data/hmdb51/HMDB51_lmdb/hmdb51_compressed_mvs.lmdb \
    --max_videos 5 \
    --frames_per_video 5 \
    --gpu_ids 0 \
    --clip_arch ViT-B/32 \
    --embed_dim 512