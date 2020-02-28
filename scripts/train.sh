#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd $root/experiments/siamrpn_r50_l234_dwxcorr_otb_8gpu
# cd $root/experiments/siamrpn_alex_dwxcorr_16gpu

# export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
# python -u $root/tools/test.py --snapshot model.pth --dataset OTB100 --config config.yaml

# python $root/tools/eval.py --tracker_path ./results --dataset OTB100 --num 1 --tracker_prefix 'model'