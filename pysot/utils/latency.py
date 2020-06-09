import torch
import numpy as np
import time
import pickle
from tqdm import tqdm
'''
stem = nn.Sequential(nn.Conv2d(3, 48, 3, padding=1, bias=False),nn.BatchNorm2d(48))
FactorizedReduce
'''
LUT={
    '1080ti':
    {

    },
    'cpu':
    {

    },
}

def compute_latency(model,input_size, iterations=None, device=None):
    # return 0.
    # compute the true latency of the model
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency
    
def preprocess_lut(filename,device):
    global LUT
    with open(filename, 'rb') as f:
        LUT=pickle.load(f)
    LUT=LUT[device]

def predict_latency(name):
    # preidict the latency based on LUT
    if name in LUT:
        return LUT[name]
    else:
        raise ValueError("Nor support LUT with {}".format(name))
# class Latency(object):
#     def __init__(self,device,ref_value):
#         self.device = device
#         self.ref_value = ref_value
#         self.table = LUT[device]
#     def predict_latency(self,op:str, in_shape, out_shape, kernel_size=0,padding=0,stride=1,affine=False):
#         ans =1.0
#         # if op == 'stem':
#         #     return table['stem '+str(in_shape[0])+' '+str(out_shape[0])]
#         # elif op == 'FactorizedReduce':
#         #     return table['FactorizedReduce '+]
#         # print(type,in_shape,out_shape,ans)
#         return ans

