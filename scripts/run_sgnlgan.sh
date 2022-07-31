python main.py \
    --network sgnlgan \
    --epoch 300 \
    --n_epochs 301 \
    --images_dir ./data/ \
    --targets_dir ./data/ \
    --conditions_dir ./data/ \
    --dataset_name SgnlGAN \
    --loss L1 \
    --sample_interval 20 \
    --checkpoint_interval 5 \
    --batch_size 1 \
    --lr 0.0002