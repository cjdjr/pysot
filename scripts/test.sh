#!/bin/bash
root='/ghome/wangmr/pysot'
export PATH=$PATH:/software/conda/envs/pysot/bin
export PYTHONPATH=$root:$PYTHONPATH

cd /gdata/wangmr/pysot/experiments/

python $root/tools/eval.py 	 \
	--tracker_path ./draw_graph \
	--dataset OTB100        \
	--num 4 		 \
	--tracker_prefix '*' \
	--vis

