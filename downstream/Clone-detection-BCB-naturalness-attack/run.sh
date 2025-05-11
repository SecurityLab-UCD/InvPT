#!/bin/bash
# run.sh 
#

docker run --name=eric-codebert-attack \
    --gpus all \
    -it \
    --mount type=bind,src=/media/zyang/codebases,dst=/workspace \
    zhouyang996/codebert-attack:v1
