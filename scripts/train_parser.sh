python main.py \
    --network human_semantic_parser \
    --epoch 17 \
    --n_epochs  25 \
    --images_dir ./data/lip/multi_person/train_LIP_A/ \
    --targets_dir ./data/lip/multi_person/LC_IDS/ \
    --dataset_name Human_Semantic_Parser_Resize \
    --loss CrossEntropy \
    --sample_interval 1000 \
    --checkpoint_interval 10 \
    --batch_size 20 \
    --lr 0.0002

