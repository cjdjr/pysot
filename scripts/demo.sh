#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
cd $root

python tools/visual.py \
    --gt_path /ghome/wangmr/pysot/testing_dataset/OTB100/Basketball/groundtruth_rect.txt \
    --path1 /gdata/wangmr/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_3/results/OTB100/checkpoint_e17/Basketball.txt \
    --path2 /gdata/wangmr/pysot/experiments/siamrpn_r18_l4_dwxcorr_otb_4gpu_1/results/OTB100/checkpoint_e20/Basketball.txt \
    --video_name /ghome/wangmr/pysot/testing_dataset/OTB100/Basketball/img