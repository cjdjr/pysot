#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_dwxcorr_otb_4gpu_1
# cd /gdata/wangmr/pysot/experiments/test
# export CUDA_VISIBLE_DEVICES=7
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/train.py --cfg config.yaml
# python -u $root/tools/test.py --snapshot model.pth --dataset OTB100 --config config.yaml

# python $root/tools/eval.py --tracker_path ./results --dataset OTB100 --num 1 --tracker_prefix 'model'