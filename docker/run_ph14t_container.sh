docker run \
    -e RUN_SCRIPT=run_sgnlgan.sh \
    -v <path_to_repo>/signlanguagegan/:/workspace \
    -v <path_to_dataset>/RWTH-PHOENIX-14T/phoenix-2014-multisigner/features/fullFrame-210x260px/train/:/workspace/data/ \
    --gpus=all \
    --ipc=host \
    --name ph2014t_container \
    --rm \
    -it env /bin/bash
    #-d env 