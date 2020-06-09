#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_3_pretrain
# cd $root/experiments/siamrpn_alex_dwxcorr_16gpu

# export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/pretrain.py --cfg config.yaml

# export CUDA_VISIBLE_DEVICES=6
# python -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --master_port=2333 \
#     $root/tools/debug_train.py --cfg config.yaml


