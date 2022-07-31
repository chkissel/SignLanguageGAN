docker run \
    -e RUN_SCRIPT=run_sgnlgan.sh \
    -v /home/ckissel/signlanguagegan/:/workspace \
    -v home/ckissel/RWTH-PHOENIX-14T/phoenix-2014-multisigner/features/fullFrame-210x260px/train/:/workspace/data/ \
    --gpus=all \
    --ipc=host \
    --name ph2014t_container \
    --rm \
    -it env /bin/bash
    #-d env 