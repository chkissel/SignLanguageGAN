docker run \
    -e RUN_SCRIPT=train_parser.sh \
    -v /home/ckissel/signlanguagegan/:/workspace \
    -v /home/ckissel/cannygan/lip:/workspace/data/lip \
    --gpus=all \
    --ipc=host \
    --name Semantic_Human_Parser_Resize \
    --rm \
    -d env
    #-it env /bin/bash
    
