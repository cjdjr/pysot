import torch
import torch.nn as nn
# from mcc_nas.utils.latency import predict_latency,compute_latency

class ReLUConvBN(nn.Module):
  """
  Stack of relu-conv-bn
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    """

    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding:
    :param affine:
    """
    super(ReLUConvBN, self).__init__()
    self.C_in=C_in
    self.C_out=C_out
    self.kernel_size=kernel_size
    self.stride=stride
    self.padding=padding
    self.affine=True
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine))

  def forward(self, x):
    return self.op(x)

  def forward_latency(self,x):
    name = "ReLUConvBN H:%d W:%d C_IN:%d C_OUT:%d KERNELSIZE:%d STRIDE:%d PADDING:%d AFFINE:%s"%(x[1],x[2],self.C_in,self.C_out,self.kernel_size,self.stride,self.padding,self.affine)
    latency = predict_latency(name)
    fun = lambda x:(x+2*self.padding-self.kernel_size)//self.stride+1
    return latency,(self.C_out,fun(x[1]),fun(x[2]))
    
  @staticmethod
  def _latency(H,W,C_in, C_out, kernel_size, stride, padding, affine=True):
    layer = ReLUConvBN(C_in, C_out, kernel_size, stride, padding, affine)
    return compute_latency(layer,(1,C_in,H,W))


class DilConv(nn.Module):
  """
  relu-dilated conv-bn
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    """

    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding: 2/4
    :param dilation: 2
    :param affine:
    """
    super(DilConv, self).__init__()
    self.C_in=C_in
    self.C_out=C_out
    self.kernel_size=kernel_size
    self.stride=stride
    self.padding=padding
    self.dilation=dilation
    self.affine=affine
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine))

  def forward(self, x):
    return self.op(x)

  def forward_latency(self,x):
    name = "DilConv H:%d W:%d C_IN:%d C_OUT:%d KERNELSIZE:%d STRIDE:%d PADDING:%d DILATION:%d AFFINE:%s"%(x[1],x[2],self.C_in,self.C_out,self.kernel_size,self.stride,self.padding,self.dilation,self.affine)
    latency = predict_latency(name)
    fun = lambda x:(x+2*self.padding-self.dilation*(self.kernel_size-1)-1)//self.stride+1
    return latency,(self.C_out,fun(x[1]),fun(x[2]))
    
  @staticmethod
  def _latency(H,W,C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    layer = DilConv(C_in, C_out, kernel_size, stride, padding, dilation, affine)
    return compute_latency(layer,(1,C_in,H,W))



class SepConv(nn.Module):
  """
  implemented separate convolution via pytorch groups parameters
  """

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    """

    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding: 1/2
    :param affine:
    """
    super(SepConv, self).__init__()
    self.C_in=C_in
    self.C_out=C_out
    self.kernel_size=kernel_size
    self.stride=stride
    self.padding=padding
    self.affine=affine
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine))

  def forward(self, x):
    return self.op(x)

  def forward_latency(self,x):
    name = "SepConv H:%d W:%d C_IN:%d C_OUT:%d KERNELSIZE:%d STRIDE:%d PADDING:%d AFFINE:%s"%(x[1],x[2],self.C_in,self.C_out,self.kernel_size,self.stride,self.padding,self.affine)
    latency = predict_latency(name)
    fun = lambda x:(x+2*self.padding-self.kernel_size)//self.stride+1
    return latency,(self.C_out,fun(x[1]),fun(x[2]))
    
  @staticmethod
  def _latency(H,W,C_in, C_out, kernel_size, stride, padding, affine=True):
    layer = SepConv(C_in, C_out, kernel_size, stride, padding, affine)
    return compute_latency(layer,(1,C_in,H,W))
  


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

  def forward_latency(self,x):
    name = "Identity H:%d W:%d C_IN:%d"%(x[1],x[2],x[0])
    latency = predict_latency(name)
    return latency,x
  
  @staticmethod
  def _latency(H,W,C_in):
    layer = Identity()
    return compute_latency(layer,(1,C_in,H,W))

class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.C_in   = C_in
    self.C_out  = C_out
    self.stride = stride
    self.is_zero = True

  def forward(self, x):
    if self.C_in == self.C_out:
      if self.stride == 1: return x.mul(0.)
      else               : return x[:,:,::self.stride,::self.stride].mul(0.)
    else:
      shape = list(x.shape)
      shape[1] = self.C_out
      zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
      return zeros

  def forward_latency(self,x):
    name = "Zero H:%d W:%d C_IN:%d C_OUT:%d STRIDE:%d"%(x[1],x[2],self.C_in,self.C_out,self.stride)
    latency = predict_latency(name)
    return latency,(self.C_out,x[1]//self.stride,x[2]//self.stride)
  
  @staticmethod
  def _latency(H,W,C_in,C_out,stride):
    layer = Zero(C_in,C_out,stride)
    return compute_latency(layer,(1,C_in,H,W))
      

class FactorizedReduce(nn.Module):
  """
  reduce feature maps height/width by half while keeping channel same
  """

  def __init__(self, C_in, C_out, affine=True):
    """

    :param C_in:
    :param C_out:
    :param affine:
    """
    super(FactorizedReduce, self).__init__()
    self.C_in=C_in
    self.C_out=C_out
    self.affine=affine
    assert C_out % 2 == 0

    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)

    # x: torch.Size([32, 32, 32, 32])
    # conv1: [b, c_out//2, d//2, d//2]
    # conv2: []
    # out: torch.Size([32, 32, 16, 16])

    out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
    out = self.bn(out)
    return out

  def forward_latency(self,x):
    name = "FactorizedReduce H:%d W:%d C_IN:%d C_OUT:%d AFFINE:%s"%(x[1],x[2],self.C_in,self.C_out,self.affine)
    latency = predict_latency(name)
    return latency,(self.C_out,x[1]//2,x[2]//2)
    
  @staticmethod
  def _latency(H,W,C_IN,C_OUT,AFFINE):
    layer = FactorizedReduce(C_IN,C_OUT,AFFINE)
    return compute_latency(layer,(1,C_IN,H,W))


class ResidualReduceBlock(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(ResidualReduceBlock, self).__init__()
    self.downsample_a = nn.Sequential(
                          ReLUConvBN(C_in,  C_out, 3, 2, 1, 1, affine),
                          ReLUConvBN(C_out, C_out, 3, 1, 1, 1, affine)
                          )
    self.downsample_b = nn.Sequential(
                          nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                          nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False)
                          )
    self.C_in  = C_in
    self.C_out = C_out

  def extra_repr(self):
    string = '{name}(inC={C_in}, outC={C_out})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs):
    downsample_a = self.downsample_a(inputs)
    downsample_b = self.downsample_b(inputs)
    
    return downsample_a + downsample_b

