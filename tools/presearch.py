# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import datetime
import math
import json
import random
import numpy as np
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.metric_logger import MetricLogger
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

from pysot.models.backbone.darts import Cell
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pysot.models.backbone.search_space import build_search_space
from pysot.models.backbone.operations import ReLUConvBN
from pysot.models.backbone.operations import FactorizedReduce
from pysot.models.backbone.operations import MixedLayer
from pysot.models.backbone.operations import Identity
from pysot.models.backbone.operations import OPS
from pysot.models.backbone.genotypes  import Genotype

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
class CIFAR10_split(torch.utils.data.Dataset):
  """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
  """
  base_folder = 'cifar-10-batches-py'
  train_list = [['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb']]

  test_list = [['test_batch', '40351d587109b95175f43aff81a1287e']]

  def __init__(self, root, split, ratio, transform=None, target_transform=None):
    assert split in ['train', 'val', 'test']
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.split = split  # training set or test set

    # now load the picked numpy arrays
    if self.split == 'test':
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()

      self.data = self.data.reshape((-1, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      self.data = []
      self.labels = []
      for fentry in self.train_list:
        f = fentry[0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        entry = pickle.load(fo, encoding='latin1')
        self.data.append(entry['data'])
        if 'labels' in entry:
          self.labels += entry['labels']
        else:
          self.labels += entry['fine_labels']
        fo.close()

      self.data = np.concatenate(self.data)
      self.data = self.data.reshape((-1, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      if self.split == 'train' and 0.0 < ratio < 1.0:
        split_size = int(np.clip(len(self.data) * ratio, 1.0, len(self.data)))
        print('using %d images from start ...' % split_size)
        # logging.getLogger('logger').info('using %d images from start ...' % split_size)
        self.data = self.data[:split_size]
        self.labels = self.labels[:split_size]
      elif self.split == 'val' and 0.0 < ratio < 1.0:
        split_size = int(np.clip(len(self.data) * ratio, 1.0, len(self.data)))
        print('using %d images from end ...' % split_size)
        # logging.getLogger('logger').info('using %d images from end ...' % split_size)
        self.data = self.data[-split_size:]
        self.labels = self.labels[-split_size:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]
    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def cifar_search_transform(is_training=True, cutout=None):
  transform_list = []
  if is_training:
    transform_list += [transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip()]

  transform_list += [transforms.ToTensor(),
                     transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                          [0.24703233, 0.24348505, 0.26158768])]

  if cutout is not None:
    transform_list += [Cutout(cutout)]

  return transforms.Compose(transform_list)

class DartsCell(nn.Module):

  def __init__(self, num_node, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, search_space):
    """
    :param num_node: 4, number of layers inside a cell
    :param multiplier: 4
    :param C_prev_prev: 48
    :param C_prev: 48
    :param C: 16
    :param reduction: indicates whether to reduce the output maps width
    :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
    in order to keep same shape between s1 and s0, we adopt prep0 layer to
    reduce the s0 width by half.
    """
    super(DartsCell, self).__init__()

    # indicating current cell is reduction or not
    self.reduction = reduction
    self.reduction_prev = reduction_prev

    # preprocess0 deal with output from prev_prev cell
    if reduction_prev:
      # if prev cell has reduced channel/double width,
      # it will reduce width by half
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, kernel_size=1,
                                    stride=1, padding=0, affine=False)
    # preprocess1 deal with output from prev cell
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    # steps inside a cell
    self.num_node = num_node  # 4
    self.multiplier = multiplier  # 4

    self.layers = nn.ModuleList()

    for i in range(self.num_node):
      # for each i inside cell, it connects with all previous output
      # plus previous two cells' output
      for j in range(2 + i):
        # for reduction cell, it will reduce the heading 2 inputs only
        stride = 2 if reduction and j < 2 else 1
        layer = MixedLayer(C, stride, op_names_list=search_space)
        self.layers.append(layer)

  def forward(self, s0, s1, weights):
    """
    :param s0:
    :param s1:
    :param weights: [14, 8]
    :return:
    """
    # print('s0:', s0.shape,end='=>')
    s0 = self.preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s0.shape, self.reduction_prev)
    # print('s1:', s1.shape,end='=>')
    s1 = self.preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]
    # print(s1.shape)

    states = [s0, s1]
    offset = 0
    # for each node, receive input from all previous intermediate nodes and s0, s1
    for i in range(self.num_node):  # 4
      # [40, 16, 32, 32]
      s = sum(self.layers[offset + j](h, weights[offset + j])
              for j, h in enumerate(states)) / len(states)
      offset += len(states)
      # append one state since s is the elem-wise addition of all output
      states.append(s)
      # print('node:',i, s.shape, self.reduction)

    # concat along dim=channel
    return torch.cat(states[-self.multiplier:], dim=1)  # 6 of [40, 16, 32, 32]


class SuperNet(nn.Module):
  """
  stack number:layer of cells and then flatten to fed a linear layer
  """

  def __init__(self,search_space, num_ch, num_cell,
               num_node=4, multiplier=4, stem_multiplier=3, num_class=10,  img_channel=3):
    """

    :param C: 16
    :param num_cell: number of cells of current network
    :param num_node: nodes num inside cell
    :param multiplier: output channel of cell = multiplier * ch
    :param stem_multiplier: output channel of stem net = stem_multiplier * ch
    :param num_class: 10
    """
    super(SuperNet, self).__init__()

    self.C = num_ch
    self.num_class    = num_class
    self.num_cell     = num_cell
    self.num_node     = num_node
    self.multiplier   = multiplier
    self.search_space = search_space

    # stem_multiplier is for stem network,
    # and multiplier is for general cell
    C_curr = stem_multiplier * num_ch  # 3*16
    # stem network, convert 3 channel to c_curr
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr))

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C_curr))


    # c_curr means a factor of the output channels of current cell
    # output channels = multiplier * c_curr
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, num_ch  # 48, 48, 16
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(num_cell):

      # for layer in the middle [1/3, 2/3], reduce via stride=2
      if i in [num_cell // 3]:
      # if i in [num_cell // 3, 2 * num_cell // 3]:
        C_curr *= 2
        reduction = True
        print(i)
      else:
        reduction = False

      # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
      # the output channels = multiplier * c_curr
      cell = DartsCell(num_node, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, search_space)
      # update reduction_prev
      reduction_prev = reduction

      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier * C_curr

    # adaptive pooling output size to 1x1
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # since cp records last cell's output channels
    # it indicates the input channel number
    self.classifier = nn.Linear(C_prev, num_class)

    # k is the total number of edges inside single cell, 14
    k = sum(1 for i in range(self.num_node) for j in range(2 + i))
    num_ops = len(self.search_space)  # 8

    self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
    self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))
    with torch.no_grad():
      # initialize to smaller value
      self.alpha_normal.mul_(1e-3)
      self.alpha_reduce.mul_(1e-3)
    self._arch_parameters = [self.alpha_normal, self.alpha_reduce]

  def forward(self, x):
    """
    in: torch.Size([b, 3, 32, 32])
    stem0: torch.Size([b, 144, 8, 8])
    stem1: torch.Size([b, 144, 8, 8])
    cell: 0 torch.Size([b, 192, 8, 8]) False
    cell: 1 torch.Size([b, 192, 8, 8]) False
    cell: 2 torch.Size([b, 384, 4, 4]) True
    cell: 3 torch.Size([b, 384, 4, 4]) False
    cell: 4 torch.Size([b, 384, 4, 4]) False
    cell: 5 torch.Size([b, 384, 4, 4]) False
    cell: 6 torch.Size([b, 384, 4, 4]) False
    cell: 7 torch.Size([b, 384, 4, 4]) False
    :param x:
    :return:
    """
    # print('in:', x.shape)
    # s0 & s1 means the last cells' output
    # s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
    s0 = self.stem0(x)
    # print('stem0 : ',s0.shape)
    s1 = self.stem1(s0)
    # print('stem:', s0.shape)
    # print('stem1 : ',s1.shape)
    
    for i, cell in enumerate(self.cells):
      if cell.reduction:  # if current cell is reduction cell
        weights = F.softmax(self.alpha_reduce, dim=-1)
      else:
        weights = F.softmax(self.alpha_normal, dim=-1)  # [14, 8]
      s0, s1 = s1, cell(s0, s1, weights)  # [40, 64, 32, 32]
    #   print(i,' : ',s1.shape)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def genotype(self):
    def _parse(weights):
      """
      :param weights: [14, 8]
      :return:
      """
      gene = []
      n = 2
      start = 0
      for i in range(self.num_node):  # for each node
        end = start + n
        W = weights[start:end].copy()  # shape=[2, 8], [3, 8], [4, 8], [5, 8]
        # i+2 is the number of connection for node i
        # sort by descending order, get strongest 2 edges
        # note here we assume the 0th op is none op, if it's not the case this will be wrong!
        edges = np.argsort(-np.max(W[:, 1:], axis=1))[:2]
        ops   = np.argmax(W[edges, 1:], axis=1) + 1
        gene += [(self.search_space[op], edge) for op, edge in zip(ops, edges)]
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

    concat = range(2 + self.num_node - self.multiplier, self.num_node + 2)
    genotype = Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)

    return genotype

def build_model():

    base_ch = cfg.PRESEARCH.BASE_CH
    cell_num = cfg.PRESEARCH.CELL_NUM
    search_space = build_search_space(cfg)
    model = SuperNet(search_space, base_ch, cell_num, num_class=10)
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

    logger.info("build dataset")
    data_dir = cfg.PRESEARCH.DATADIR
    train_transform = cifar_search_transform(is_training=True)
    val_transform   = cifar_search_transform(is_training=False)
    train_dataset = CIFAR10_split(root=data_dir, 
                                  split='train', 
                                  ratio=0.5,
                                  transform=train_transform
                                  )
    val_dataset   = CIFAR10_split(root=data_dir, 
                                  split='val', 
                                  ratio=0.5,
                                  transform=val_transform
                                  )

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

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True)
    # print(len(val_dataset))
    # print(len(val_loader))
    # print(val_dataset[2])
    # print(val_dataset[49999])
    # print(train_dataset[0])
    return train_loader,val_loader


def build_optimizer(weights, alphas):

    w_lr = cfg.PRESEARCH.W_LEARNING_RATE
    a_lr = cfg.PRESEARCH.A_LEARNING_RATE
    w_wd = cfg.PRESEARCH.W_WEIGHT_DECAY
    a_wd = cfg.PRESEARCH.A_WEIGHT_DECAY
    w_momentum = cfg.PRESEARCH.W_MOMENTUM
    
    w_optimizer = torch.optim.SGD(weights, w_lr, momentum=w_momentum, weight_decay=w_wd)
    a_optimizer = torch.optim.Adam(alphas, lr=a_lr, betas=(0.5, 0.999), weight_decay=a_wd)

    return w_optimizer, a_optimizer



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


def train(train_loader, val_loader, model,tb_writer):




    rank = get_rank()
    weights = [v for k, v in model.named_parameters() if 'alpha' not in k]
    alphas  = [v for k, v in model.named_parameters() if 'alpha' in k]
    w_optimizer, a_optimizer = build_optimizer(weights, alphas)
    criterion = nn.CrossEntropyLoss().cuda()
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, cfg.TRAIN.EPOCH, eta_min=cfg.PRESEARCH.W_MIN_LEARNING_RATE)


    # Start training
    logger.info('Start Presearch')
    meters = MetricLogger(delimiter="  ")

    start_epoch = 0
    total_epoch = cfg.TRAIN.EPOCH

    alpha_list = []
    best_accuracy = -1
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.step(epoch)
        # train_sampler.set_epoch(epoch)
        # val_sampler.set_epoch(epoch)
        alpha_list.append([])
        steps_per_epoch = len(train_loader)

        model.train()
        end = time.time()
        for step, ((w_inputs, w_targets), (a_inputs, a_targets)) in enumerate(zip(train_loader, val_loader)):
            w_inputs, w_targets = w_inputs.cuda(), w_targets.cuda(non_blocking=True)
            a_inputs, a_targets = a_inputs.cuda(), a_targets.cuda(non_blocking=True)

            # Measure data loading time
            data_time = time.time() - end

            # Update weights
            w_optimizer.zero_grad()
            w_outputs = model(w_inputs)
            w_loss    = criterion(w_outputs, w_targets)
            w_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            w_optimizer.step()

            # Update alpha
            if epoch > cfg.PRESEARCH.EPOCH_START_ARCH_UPDATE:
                # with torch.no_grad():
                a_optimizer.zero_grad()
                # Only support 1st order update in this re-implementation
                a_outputs = model(a_inputs)
                a_loss    = criterion(a_outputs, a_targets)
                a_loss.backward()
                a_optimizer.step()
            else:
                a_loss = torch.tensor([0]).cuda()

            normal_alphas = model.module.alpha_normal.detach().cpu().numpy()
            reduce_alphas = model.module.alpha_reduce.detach().cpu().numpy()

            single_alpha_dict = dict(normal_alphas=normal_alphas,
                                    reduce_alphas=reduce_alphas,
                                    )
            alpha_list[-1].append(single_alpha_dict)  
            
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            # Update meters
            meters.update(train_loss = w_loss,
                            val_loss   = a_loss)

            remain_steps = (total_epoch-1-epoch) * steps_per_epoch + (steps_per_epoch-1-step)
            eta_seconds  = meters.time.global_avg * remain_steps
            eta_string   = str(datetime.timedelta(seconds=int(eta_seconds)))

            if step % 10 == 0:
                logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                'epoch: {epoch} '
                                "step: {step}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            epoch=epoch,
                            step=step,
                            meters=str(meters),
                            lr=min(w_scheduler.get_lr()),
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                )
            # break

        # Get @1 accuracy on validation set
        model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                outputs = model(inputs)
                total_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().item()

        acc = 100. * correct / len(val_loader.dataset)
        total_loss = total_loss / len(val_loader)
        logger.info('Validation loss after epoch[%d]==> %.5f Precision@1 ==> %.2f%% \n' % (epoch, total_loss, acc))
        
        logger.info('Genotype after epoch[{}]'.format(model.module.genotype()))

        # alphas_save_pattern   = os.path.join(cfg.OUTPUT_DIR, 'seed-{}-alpha.pth')
        # models_save_pattern   = os.path.join(cfg.OUTPUT_DIR, 'seed-{}-model.pth')
        # genotype_save_pattern = os.path.join(cfg.OUTPUT_DIR, 'seed-{}-genotype.pth')

        # torch.save(alpha_list, alphas_save_pattern.format(rand_seed))
        # torch.save(model.state_dict(), models_save_pattern.format(rand_seed))
        # torch.save(model.module.genotype(), genotype_save_pattern.format(rand_seed))

        # check the best accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            logger.info('Update best genotypes as:')
            logger.info(model.module.genotype())
            # torch.save(single_alpha_dict, alphas_save_pattern.format('{}-best'.format(rand_seed)))
            torch.save(model.state_dict(), cfg.PRESEARCH.MODEL_SAVE_DIR)
            # torch.save(model.module.genotype(), genotype_save_pattern.format('{}-best'.format(rand_seed)))



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

    # build dataset loader
    train_loader,val_loader = build_data_loader()

    # for step, (inputs, targets) in enumerate(val_loader):
    #     print(step)
    #     print(inputs)
    #     print(targets)
    # return

    # create model
    model = build_model().cuda().train()
    # return
    # model(torch.randn(1,3,32,32).cuda())
    # return
    # print(model.backbone)
    

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




    # logger.info("total train_loader "+str(len(train_loader)))
    # logger.info("totatl val_loader "+str(len(val_loader)))

    # for step, (inputs, targets) in enumerate(val_loader):
    #     logger.info(str(step))
    # return




    # build optimizer and lr_scheduler
    # optimizer, lr_scheduler = build_opt_lr(model)

    # resume training
    # if cfg.TRAIN.RESUME:
    #     logger.info("resume from {}".format(cfg.TRAIN.RESUME))
    #     assert os.path.isfile(cfg.TRAIN.RESUME), \
    #         '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
    #     model, optimizer, cfg.TRAIN.START_EPOCH = \
    #         restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # # load pretrain
    # elif cfg.TRAIN.PRETRAINED:
    #     load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = DistModule(model)
    # return
    # logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, val_loader,dist_model,tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

