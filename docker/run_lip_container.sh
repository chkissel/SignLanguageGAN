docker run \
    -e RUN_SCRIPT=train_parser.sh \
    -v <path_to_repo>/signlanguagegan/:/workspace \
    -v <path_to_dataset>/cannygan/lip:/workspace/data/lip \
    --gpus=all \
    --ipc=host \
    --name Semantic_Human_Parser_Resize \
    --rm \
    -d env
    #-it env /bin/bash
    
