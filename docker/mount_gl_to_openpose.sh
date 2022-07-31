docker run \
    -v /home/ckissel/gebaerdenlernen/gebaerdenlernen/features/frames/:/openpose/data \
    -v /home/ckissel/signlanguagegan/scripts/openpose_iterator.sh:/openpose/openpose_iterator.sh \
    --gpus=all \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    --name msasl_container \
    --rm \
    -it \
    -d cwaffles/openpose 