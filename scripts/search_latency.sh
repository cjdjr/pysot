#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_latency_search_dwxcorr_otb_4gpu
# cd /gdata/wangmr/pysot/experiments/debug
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/search_latency.py --cfg config.yaml
# python -u $root/tools/test.py --snapshot model.pth --dataset OTB100 --config config.yaml

# python $root/tools/eval.py --tracker_path ./results --dataset OTB100 --num 1 --tracker_prefix 'model'