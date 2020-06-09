# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

from pysot.models.backbone.darts import Cell
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, kernel_size=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, kernel_size=2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    # print("x.view(x.size(0), -1) = ",x.view(x.size(0), -1).shape)
    x = self.classifier(x.view(x.size(0), -1))
    return x

class NetworkImageNet(nn.Module):

  def __init__(self, genotype, C, layers,auxiliary, num_classes=1000):

    super(NetworkImageNet, self).__init__()
    self.drop_path_prob = 0.0
    self._layers = layers

    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C))

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C))

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(28)
    # print('C_prev : ',C_prev)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    """
    in: torch.Size([b, 3, 255, 255])
    stem0: torch.Size([b, 48, 64, 64])
    stem1: torch.Size([b, 48, 64, 64])
    cell: 0 torch.Size([b, 192, 64, 64]) False
    cell: 1 torch.Size([b, 192, 64, 64]) False
    cell: 2 torch.Size([b, 192, 64, 64]) False
    cell: 3 torch.Size([b, 384, 32, 32]) True
    cell: 4 torch.Size([b, 384, 32, 32]) False
    cell: 5 torch.Size([b, 384, 32, 32]) False
    cell: 6 torch.Size([b, 384, 32, 32]) False
    cell: 7 torch.Size([b, 384, 32, 32]) False
    cell: 8 torch.Size([b, 384, 32, 32]) False
    cell: 9 torch.Size([b, 384, 32, 32]) False
    :param x:
    :return:
    """
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    #   print(i,' : ',s1.shape)
      if i == self._layers // 3:
        if self._auxiliary and self.training:
        #   print("size(s1) : ",s1.shape)
          logits_aux = self.auxiliary_head(s1)
    # print("size(s1) : ",s1.shape)
    out = self.global_pooling(s1)
    # print("size(out) : ",out.shape)
    # print('out.view(out.size(0), -1) : ',out.view(out.size(0), -1).shape)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

def build_model():
    genotype = cfg.PRETRAIN.GENOTYPE
    base_ch = cfg.PRETRAIN.BASE_CH
    cell_num = cfg.PRETRAIN.CELL_NUM
    auxiliary = cfg.PRETRAIN.AUXILIARY
    model = NetworkImageNet(genotype, base_ch, cell_num, auxiliary, num_classes=1000)
    return model

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_data_loader():

    class my_dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.L = 256
            self.img = [np.random.randn(3,224,224).astype(np.float32) for i in range(self.L)]

        def __getitem__(self,index):
            label = 0
            return self.img[index],label

        def __len__(self):
            return self.L

    logger.info("build train dataset")
    # train_dataset
    traindir = os.path.join(cfg.PRETRAIN.DATADIR,'train')
    valdir = os.path.join(cfg.PRETRAIN.DATADIR,'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    # train_dataset = my_dataset()




    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              sampler=train_sampler)
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    # val_dataset = my_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
    # print(len(val_dataset))
    # print(len(val_loader))
    # print(val_dataset[2])
    # print(val_dataset[49999])

    return train_loader,val_loader


def build_opt_lr(model):


    optimizer = torch.optim.SGD(model.parameters(), cfg.TRAIN.BASE_LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # print(cfg.TRAIN.EPOCH)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


def train(train_loader, val_loader, model, optimizer, lr_scheduler, tb_writer):

    best_acc_top1 = -1

    def validate(epoch):
        model.eval()
        top1 = 0
        top5 = 0
        correct = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(val_loader):
                # logger.info(str(step))
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                # ??????????????????????s?
                outputs,_ = model(inputs)
                # measure accuracy and record loss
                _, pred = outputs.data.topk(5, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
                top5 += correct[:5].view(-1).float().sum(0, keepdim=True).item()
                total_loss += criterion(outputs, targets).item()
                # _, predicted = torch.max(outputs.data, 1)
                # correct += predicted.eq(targets.data).cpu().sum().item()

        top1 *= 100 / len(val_loader.dataset)
        top5 *= 100 / len(val_loader.dataset)
        total_loss = total_loss / len(val_loader)
        logger.info('Validation loss after epoch[%d]==> %.5f Precision@1 ==> %.2f%% Precision@5 ==> %.2f%% \n' % (epoch, total_loss, top1,top5))
        # final_models_save_pattern   = os.path.join(cfg.OUTPUT_DIR, 'seed-{}-final_model.pth')
        # torch.save(model.state_dict(), final_models_save_pattern.format(rand_seed))
        nonlocal best_acc_top1
        if top1 > best_acc_top1:
            best_acc_top1 = top1
            logger.info('Update best model !')
            torch.save(model.state_dict(), cfg.PRETRAIN.MODEL_SAVE_DIR)
        logger.info('The current best accuracy is %.5f \n',best_acc_top1)

    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()
    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    # num_per_epoch = len(train_loader.dataset) // \
    #     cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    total_epoch = cfg.TRAIN.EPOCH
    # epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    # logger.info("model\n{}".format(describe(model.module)))
    for epoch in range(start_epoch,total_epoch):
        logger.info('epoch {} lr {}'.format(epoch+1, cur_lr))
        steps_per_epoch = len(train_loader)
        model.train()
        end = time.time()
        for step,(inputs,targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # Measure data loading time
            data_time = average_reduce(time.time() - end)

            outputs,_ = model(inputs)

            loss = criterion(outputs,targets)
            if step % 100 == 0 :
                logger.info("step : {}   loss : {}".format(step,loss))

            if is_valid_number(loss.data.item()):
                optimizer.zero_grad()
                loss.backward()
                reduce_gradients(model)
                # clip gradient
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()

            # break
        
        validate(epoch)
        lr_scheduler.step(epoch)
        cur_lr = lr_scheduler.get_cur_lr()



def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        # logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = build_model().cuda().train()
    # model(torch.randn(1,3,224,224).cuda())
    # print(model.backbone)
    # return 

    # load pretrained backbone weights
    # if cfg.BACKBONE.PRETRAINED:
    #     cur_path = os.path.dirname(os.path.realpath(__file__))
    #     backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
    #     load_pretrain(model.backbone, backbone_path)
    # print("ok")
    # return
    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader,val_loader = build_data_loader()


    logger.info("total train_loader "+str(len(train_loader)))
    logger.info("totatl val_loader "+str(len(val_loader)))

    # for step, (inputs, targets) in enumerate(val_loader):
    #     logger.info(str(step))
    # return




    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = DistModule(model)
    # return
    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, val_loader,dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

