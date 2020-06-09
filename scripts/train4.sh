#!/bin/bash
export PATH=$PATH:/software/conda/envs/pysot/bin

root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_4
# cd $root/experiments/siamrpn_alex_dwxcorr_16gpu

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=2333 \
    $root/tools/train.py --cfg config.yaml

cd /gdata/wangmr/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_4

START=16
END=20
seq $START 1 $END | \
    xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
    xargs -I {} python -u $root/tools/test.py --snapshot {} --config config.yaml --dataset OTB100 2>&1 | tee logs/test_dataset.log

# python -u $root/tools/test.py --snapshot snapshot/checkpoint_e20.pth --dataset OTB100 --config config.yaml
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_4
python $root/tools/eval.py --tracker_path ./results --dataset OTB100 --num 4 --tracker_prefix 'ch*' | tee -a logs/test_dataset.log

