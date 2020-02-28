#!/bin/bash
root='/ghome/wangmr/pysot'
export PYTHONPATH=$root:$PYTHONPATH

cd $root/experiments/siamrpn_alex_dwxcorr_otb
python -u $root/tools/test.py --snapshot model.pth --dataset OTB100 --config config.yaml

python $root/tools/eval.py --tracker_path ./results --dataset OTB100 --num 1 --tracker_prefix 'model'