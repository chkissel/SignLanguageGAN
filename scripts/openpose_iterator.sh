# docker exec -it -d <docker_container_ID> /bin/bash ./openpose_iterator.sh

for d in ./data/*
do
    if [ ! -d "$d/poses/" ] 
    then
        (mkdir "$d/poses/" && \
        ./build/examples/openpose/openpose.bin \
        --image_dir "$d/1/" \
        --face --hand \
        --display 0 \
        --disable_blending \
        --write_images "$d/poses/" \
        --write_json "$d/poses/"
    )
    else
        echo "Error: Directory $d/poses/ exists." 
    fi
done

