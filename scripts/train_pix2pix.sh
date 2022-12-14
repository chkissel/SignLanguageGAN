python main.py \
    --network pix2pix \
    --epoch 0 \
    --n_epochs 3 \
    --images_dir ./data/ \
    --targets_dir ./data/ \
    --conditions_dir ./data/ \
    --dataset_name pix2pix \
    --loss L1 \
    --sample_interval 500 \
    --checkpoint_interval 20 \
    --batch_size 5 \
    --lr 0.0002