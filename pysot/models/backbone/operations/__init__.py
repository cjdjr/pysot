import torch
import torch.nn as nn

from .basic_operations import ReLUConvBN
from .basic_operations import SepConv
from .basic_operations import DilConv
from .basic_operations import Identity
from .basic_operations import Zero
from .basic_operations import FactorizedReduce
from .basic_operations import ResidualReduceBlock


__all__ = ["ReLUConvBN", "SepConv", "DilConv", "ResidualReduceBlock", "Identity", "Zero", "FactorizedReduce"]

OPS = {
  'none'         : lambda C_in, C_out, stride, affine: Zero(C_in, C_out, stride),
  'skip_connect' : lambda C_in, C_out, stride, affine: Identity() if stride ==1 else FactorizedReduce(C_in, C_out, affine=affine),
  # spatial pooling
  'avg_pool_3x3': lambda C_in, C_out, stride, affine: nn.AvgPool2d(3, stride=stride,
                                                              padding=1, count_include_pad=False),
  'max_pool_3x3': lambda C_in, C_out, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  # separable relu-conv-bn operations
  'sep_conv_3x3' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine),
  # dilation relu-conv-bn operations
  'dil_conv_3x3' : lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine=affine),
  # normal relu-conv-bn operations
  'nor_conv_1x1' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 1, stride, 0, affine=affine),
  'nor_conv_3x3' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 3, stride, 1, affine=affine),
  'nor_conv_5x5' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 5, stride, 2, affine=affine),
  'nor_conv_7x7' : lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 7, stride, 3, affine=affine),
  # separate convlution operations
  'conv_7x1_1x7': lambda C_in, C_out, stride, affine: nn.Sequential(
                            nn.ReLU(inplace=False),
                            nn.Conv2d(C_in, C_out, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
                            nn.Conv2d(C_in, C_out, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
                            nn.BatchNorm2d(C_out, affine=affine)),
  }


class MixedLayer(nn.Module):
  def __init__(self, c, stride, op_names_list):
    super(MixedLayer, self).__init__()
    self.op_names_list = op_names_list
    self.layers = nn.ModuleList()
    """
    PRIMITIVES = [
                'none',
                'max_pool_3x3',
                'avg_pool_3x3',
                'skip_connect',
                'sep_conv_3x3',
                'sep_conv_5x5',
                'dil_conv_3x3',
                'dil_conv_5x5'
            ]
    """
    for op_name in op_names_list:
      layer = OPS[op_name](c, c, stride, False)
      if 'pool' in op_name:
        layer = nn.Sequential(layer, nn.BatchNorm2d(c, affine=False))

      self.layers.append(layer)

  def forward(self, x, weights):
    return sum([w * layer(x) for w, layer in zip(weights, self.layers)])

  def forward_latency(self,x,weights):
    latency = 0.
    y=None
    for i in range(len(weights)):
      if 'pool' not in self.op_names_list[i]:
        oplatency,y=self.layers[i].forward_latency(x)
        latency += oplatency * weights[i]
    return latency,y
