docker run \
    -v <path_to_dataset>/ms-asl/MS-ASL/MS-ASL/features/gloss-level/train/:/openpose/data \
    -v <path_to_repo>/signlanguagegan/scripts/openpose_iterator.sh:/openpose/openpose_iterator.sh \
    --gpus=all \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    --name msasl_container \
    --rm \
    -it \
    -d cwaffles/openpose 