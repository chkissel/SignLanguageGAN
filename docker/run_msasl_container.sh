docker run \
    -e RUN_SCRIPT=run_sgnlgan.sh \
    -v <path_to_dataset>/ms-asl/MS-ASL/MS-ASL/features/gloss-level/train/:/workspace \
    --gpus=all \
    --ipc=host \
    --name msasl_container \
    --rm \
    -it env /bin/bash
    #-d env 