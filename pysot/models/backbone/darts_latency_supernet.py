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
from pysot.core.config import cfg
from .search_space import build_search_space
from pysot.utils.latency import predict_latency,compute_latency

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

  def forward_latency(self, s0, s1, weights):
    """
    :param s0:
    :param s1:
    :param weights: [14, 8]
    :return:
    """
    latency = 0.

    oplatency,s0 = self.preprocess0.forward_latency(s0) 
    latency+=oplatency

    oplatency,s1 = self.preprocess1.forward_latency(s1)
    latency+=oplatency

    states = [s0, s1]
    offset = 0
    # for each node, receive input from all previous intermediate nodes and s0, s1
    for i in range(self.num_node):  # 4
      # [40, 16, 32, 32]
      for j,h in enumerate(states):
        oplatency,s=self.layers[offset+j].forward_latency(h,weights[offset+j])
        latency+=oplatency
      offset+=len(states)
      states.append(s)

    channel=sum(tmp[0] for tmp in states[-self.multiplier:])
    return latency,(channel,states[-1][1],states[-1][2])

class SuperNet(nn.Module):
  """
  stack number:layer of cells and then flatten to fed a linear layer
  """

  def __init__(self, used_layers,search_space, num_ch, num_cell,
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

    self.used_layers=used_layers
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
    # self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # since cp records last cell's output channels
    # it indicates the input channel number
    # self.classifier = nn.Linear(C_prev, num_class)

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
    in: torch.Size([b, 3, 255, 255])
    stem0: torch.Size([b, 144, 64, 64])
    stem1: torch.Size([b, 144, 64, 64])
    cell: 0 torch.Size([b, 192, 64, 64]) False
    cell: 1 torch.Size([b, 192, 64, 64]) False
    cell: 2 torch.Size([b, 384, 32, 32]) True
    cell: 3 torch.Size([b, 384, 32, 32]) False
    cell: 4 torch.Size([b, 384, 32, 32]) False
    cell: 5 torch.Size([b, 384, 32, 32]) False
    cell: 6 torch.Size([b, 384, 32, 32]) False
    cell: 7 torch.Size([b, 384, 32, 32]) False
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
    out=[]
    for i, cell in enumerate(self.cells):
      if cell.reduction:  # if current cell is reduction cell
        weights = F.softmax(self.alpha_reduce, dim=-1)
      else:
        weights = F.softmax(self.alpha_normal, dim=-1)  # [14, 8]
      s0, s1 = s1, cell(s0, s1, weights)  # [40, 64, 32, 32]
      if i in self.used_layers:
        out.append(s1)
    #   print(i,' : ',s1.shape)

    # print("len out : ",len(out))
    # print("used_layers : ",self.used_layers)
    if len(out)==1:
      return out[0]
    else:
      return out

  def forward_latency(self,x):
    # todo : stem_forward_latency
    latency = 0.
    s0=(144,64,64)
    s1=s0
    # print("after stem shape : ",s0)
    # cell_forward_latency
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alpha_reduce, dim=-1)
      else:
        weights = F.softmax(self.alpha_normal, dim=-1)
      # execute cell() firstly and then assign s0=s1, s1=result
      cell_latency,s = cell.forward_latency(s0,s1,weights)
      latency+=cell_latency
      s0,s1 = s1,s
      # print('cell:',i, s1, cell.reduction, cell.reduction_prev,cell_latency.item())
    return latency


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

def darts_latency_supernet(**kwargs):
    """Constructs a Darts_latency_supernet model.

    """
    # model = NetworkCIFAR(**kwargs)
    kwargs['search_space']=build_search_space(cfg)
    model = SuperNet(**kwargs)
    return model
