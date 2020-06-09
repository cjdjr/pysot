#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_latency_search_dwxcorr_otb_4gpu

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=2333 \
    $root/tools/search_latency.py --cfg config.yaml
