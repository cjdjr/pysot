# PySOT-NAS

## 环境安装

推荐使用docker，运行下面指令：

```bash
docker pull d243100603/pysot
```



## 数据集准备

### 训练集

* VID
* YOUTUBEBB
* DET
* COCO

对于每个数据集如何准备，在 [training_dataset](training_dataset) 目录中分别介绍。

### 测试集准备

将测试集放在`testing_dataset`目录下。



## 搜索backbone

### 直接在上述训练集上搜索

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
root=/path/to/pysot
export PYTHONPATH=$root:$PYTHONPATH
cd /path/to/pysot/experiments/siamrpn_darts_search_dwxcorr_otb_4gpu_1
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/search.py --cfg config.yaml
```

可以参考`pysot/scripts/search.sh`的写法



### 预搜索



```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
root=/path/to/pysot
export PYTHONPATH=$root:$PYTHONPATH
cd /path/to/pysot/experiments/siamrpn_darts_presearch_1
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=2333 \
    $root/tools/presearch.py --cfg config.yaml
```

得到的预搜索模型将保存在`/pretrained_models`目录下

可以参考`pysot/scripts/presearch.sh`的写法



### 附加latency的搜索

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
root=/path/to/pysot
export PYTHONPATH=$root:$PYTHONPATH
cd /path/to/pysot/experiments/siamrpn_darts_latency_search_dwxcorr_otb_4gpu
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/search_latency.py --cfg config.yaml
```

可以参考`pysot/scripts/search_latency.sh`的写法。



## 评估搜索结果

### 没有pretrain

1、需要手动将搜索出来的Genotype写入到config文件中

2、

```bash
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
root=/path/to/pysot
export PYTHONPATH=$root:$PYTHONPATH
cd /path/to/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_1
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/train.py --cfg config.yaml
START=10
END=20
seq $START 1 $END | \
    xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
    xargs -I {} python -u $root/tools/test.py --snapshot {} --config config.yaml --dataset OTB100 2>&1 | tee logs/test_dataset.log

# python -u $root/tools/test.py --snapshot snapshot/checkpoint_e20.pth --dataset OTB100 --config config.yaml
cd /gdata/wangmr/pysot/experiments/siamrpn_darts_eval_dwxcorr_otb_4gpu_1
python $root/tools/eval.py --tracker_path ./results --dataset OTB100 --num 4 --tracker_prefix 'ch*' | tee -a logs/test_dataset.log
```

可以参考`pysot/scripts/train_r18.sh`的写法。



### pretrain

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    $root/tools/pretrain.py --cfg config.yaml
```

得到的pretrain模型将保存在`/pretrained_models`目录下

可以参考`pysot/scripts/pretrain.sh`的写法



## TODO

* 正在测试搜索的结果在imgnet上pretrain之后在大的数据集上train之后的结果如何
* 修改搜索空间，将搜索空间改成zoomed-conv
* 利用gumbel-softmax trick 将显存占用降下来，从而可以在imgnet上预搜索