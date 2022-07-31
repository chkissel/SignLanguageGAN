docker run \
    -e RUN_SCRIPT=run_sgnlgan.sh \
    -v /Users/Chris/Documents/Code/bundestagGAN/:/workspace \
    --gpus=all \
    --ipc=host \
    --name msasl_container \
    --rm \
    -it env /bin/bash
    #-d env 