docker run \
    -e RUN_SCRIPT=run_sgnlgan.sh \
    -v /home/ckissel/signlanguagegan/:/workspace \
    -v /home/ckissel/gebaerdenlernen/gebaerdenlernen/:/workspace/data/ \
    --gpus=all \
    --ipc=host \
    --name gl_container \
    --rm \
    -it env /bin/bash
