docker run \
    -v /home/ckissel/RWTH-PHOENIX-14T/phoenix-2014-multisigner/features/fullFrame-210x260px/train/:/openpose/data \
    -v /home/ckissel/signlanguagegan/scripts/openpose_iterator.sh:/openpose/openpose_iterator.sh \
    --gpus=all \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    --name ph2014T_container \
    --rm \
    -it \
    -d cwaffles/openpose 
    #cwaffles/openpose /bin/bash
    