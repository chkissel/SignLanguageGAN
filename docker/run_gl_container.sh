docker run \
    -e RUN_SCRIPT=run_sgnlgan.sh \
    -v <path_to_repo>/:/workspace \
    -v <path_to_dataset>/gebaerdenlernen/gebaerdenlernen/:/workspace/data/ \
    --gpus=all \
    --ipc=host \
    --name gl_container \
    --rm \
    -it env /bin/bash
