#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

NGPUS=8
CFGFILEPATH=./configs/forgery2mask/forgery2mask_convnext_atto_ffiw.py
PORT=${PORT:-5667}
NNODES=${NNODES:-1}
NODERANK=${NODERANK:-0}
MASTERADDR=${MASTERADDR:-"127.0.0.1"}
TORCHVERSION=`python -c 'import torch; print(torch.__version__)'`


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 \
torchrun --nnodes=$NNODES \
        --nproc_per_node=$NGPUS \
        --master_addr=$MASTERADDR \
        --master_port=$PORT \
        --node_rank=$NODERANK \
        main/train.py --nproc_per_node $NGPUS \
                        --cfgfilepath $CFGFILEPATH ${@:3} 
        
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 \
torchrun --nnodes=$NNODES --nproc_per_node=$NGPUS --master_addr=$MASTERADDR --master_port=$PORT --node_rank=$NODERANK \
    main/test.py --nproc_per_node $NGPUS --cfgfilepath $CFGFILEPATH ${@:3}