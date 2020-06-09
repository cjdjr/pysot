#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=7
cd $root
python $root/tools/run_latency.py --filename look_up_table.pkl --device 1080ti