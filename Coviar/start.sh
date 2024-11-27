python train.py --lr 0.005 --batch-size 2 --arch mamba --num_segments 8 \
 	--data-name hmdb51 --representation iframe \
    --data-root /datasets/hmdb51/mpeg4_videos \
    --train-list data/datalists/hmdb51_split1_train.txt \
    --test-list data/datalists/hmdb51_split1_test.txt \
 	--model-prefix ./exp/hmdb51_mamba_model_I_Prame \
 	--lr-steps 5 10 20 40 --epochs 120 \
 	--gpus 2 3 4 5
