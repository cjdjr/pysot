#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd $root/experiments/siamrpn_darts_dwxcorr_otb_4gpu
# cd $root/experiments/siamrpn_alex_dwxcorr_16gpu

# export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
