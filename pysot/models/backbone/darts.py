import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import inspect
import os
from .genotypes  import Genotype
from .operations import ReLUConvBN
from .operations import FactorizedReduce
from .operations import MixedLayer
from .operations import Identity
from .operations import OPS

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    #print("enter drop_path ! ")
    keep_prob = 1. - drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    """
    :param genotype:
    :param C_prev_prev:
    :param C_prev:
    :param C:
    :param reduction:
    :param reduction_prev:
    """
    super(Cell, self).__init__()
    # cur_path = os.path.dirname(os.path.realpath(__file__))
    # genotype_path = os.path.join(cur_path, '../../../', genotype)
    # # print(genotype_path)
    # genotype = torch.load(genotype_path)
    # print(C_prev_prev, C_prev, C)

    genotype = eval(genotype)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, kernel_size=1, stride=1, padding=0)
    self.preprocess1 = ReLUConvBN(C_prev, C, kernel_size=1, stride=1, padding=0)

    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat

    assert len(op_names) == len(indices)

    self._num_nodes = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    #print(OPS)
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      #print(name,' ',C,' ',stride)
      #print(OPS[name])
      #fun=lambda C, stride, affine: C+stride
      #fun=OPS['sep_conv_3x3']
      #print("ce ",fun(C,stride,affine=True))
      #name = 'none'
      #print(inspect.getargspec(OPS[name]))
      op = (OPS[name])(C,C, stride, affine=True)
      #print("okkkkkkkkkkkkkkkk")
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    """

    :param s0:
    :param s1:
    :param drop_prob:
    :return:
    """
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._num_nodes):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)

      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)

      s = (h1 + h2) / 2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class NetworkCIFAR(nn.Module):
  def __init__(self, genotype, C, layers):
    super(NetworkCIFAR, self).__init__()
    self.drop_path_prob = 0.0
    self._layers = layers
    # self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(nn.Conv2d(3, C_curr, kernel_size=3, padding=1,bias=False), # stride=1
                              nn.BatchNorm2d(C_curr))

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    # if auxiliary:
    #   self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    # self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    # logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      print(i,' : ',s1.shape)
    #   if i == 2 * self._layers // 3:
    #     if self._auxiliary and self.training:
    #       logits_aux = self.auxiliary_head(s1)
    # out = self.global_pooling(s1)
    # logits = self.classifier(out.view(out.size(0), -1))
    # return logits
    return None


class NetworkImageNet(nn.Module):

  def __init__(self, used_layers, genotype, C, layers):
    super(NetworkImageNet, self).__init__()
    self.drop_path_prob = 0.0
    self._layers = layers
    self.used_layers = used_layers
    # self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C))

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C))

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
    #   if i == 2 * layers // 3:
    #     C_to_auxiliary = C_prev

    # if auxiliary:
    #   self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    # self.global_pooling = nn.AvgPool2d(7)
    # self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    # logits_aux = None
    s0 = self.stem0(input)
    # print('stem0 : ',s0.shape)
    s1 = self.stem1(s0)
    # print('stem1 : ',s1.shape)
    out=[]
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i in self.used_layers:
        out.append(s1)
      # print(i,' : ',s1.shape)
      
    if len(out)==1:
      return out[0]
    else:
      return out


def darts(**kwargs):
    """Constructs a Darts model.

    """
    # model = NetworkCIFAR(**kwargs)
    model = NetworkImageNet(**kwargs)
    return model

# if __name__ == '__main__':
#   import os
#   import pickle
#   from genotypes import *
#   from utils.utils import count_flops, count_parameters
#
#   os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#   os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
#
#   def hook(self, input, output):
#     print(output.data.cpu().numpy().shape)
#     pass
#
#
#   genotype = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1),
#                               ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
#                               ('sep_conv_3x3', 0), ('sep_conv_5x5', 2),
#                               ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)],
#                       normal_concat=range(2, 6),
#                       reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
#                               ('sep_conv_3x3', 0), ('avg_pool_3x3', 1),
#                               ('dil_conv_3x3', 3), ('sep_conv_3x3', 0),
#                               ('avg_pool_3x3', 1), ('max_pool_3x3', 0)],
#                       reduce_concat=range(2, 6))
#
#   net = NetworkCIFAR(genotype=genotype, C=36, layers=20, auxiliary=0.4, num_classes=10)
#
#   for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#       m.register_forward_hook(hook)
#
#   y = net(torch.randn(2, 3, 32, 32))
#   print(y[0].size())
#
#   count_parameters(net)
#   count_flops(net, input_size=32)
