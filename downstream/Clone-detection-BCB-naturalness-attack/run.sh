#!/bin/bash
# run.sh 
#

# Reset
docker rm eric-codebert-attack

docker run --name=eric-codebert-attack \
    --gpus all \
    -it \
    --mount type=bind,src=.,dst=/workspace \
    zhouyang996/codebert-attack:v1
