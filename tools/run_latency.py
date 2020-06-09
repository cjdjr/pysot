from pysot.models.backbone.darts_supernet import SuperNet
from pysot.models.backbone.operations import Identity,Zero,ReLUConvBN,DilConv,SepConv,FactorizedReduce
import pickle
import argparse
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir  = "{}/..".format(this_dir)

def parse_args():
  parser = argparse.ArgumentParser(description="PyTorch Neural Architecture Search")
  parser.add_argument(
        "--device",
        default="1080ti",
        type=str,
  )
  parser.add_argument(
        "--filename",
        default="",
        type=str,
  )  
  args = parser.parse_args()

  return args

def main():

    args = parse_args()
    device = args.device
    filename = lib_dir+"/"+args.filename

    print(filename)

    LUT={}
    LUT[device]={}
    # with open(filename, 'rb') as f:
    #     LUT=pickle.load(f)
    #     print(LUT)
    # return
    Fun=[
        'Stem',
        'ReLUConvBN',
        'DilConv',
        'SepConv',
        'Identity',
        'Zero',
        'FactorizedReduce',
        'Tail',
    ]
    for op in Fun:
        if op=='Stem':
            pass
            # Hset=[255]
            # Wset=[255]
            # for H in Hset:
            #     for W in Wset:
            #         latency = SuperNet._stem_latency(H,W)
            #         name=op+" H:%d"%H+" W:%d"%W
            #         LUT[device][name]=latency
        elif op=='Tail':
            pass
        elif op=='Identity':
            Hset=[64,32]
            Wset=[64,32]
            C_INset=[48,96]
            for i in range(len(Hset)):
                H=Hset[i]
                W=Wset[i]
                C_IN=C_INset[i]

                latency = Identity._latency(H,W,C_IN)
                name=op+" H:%d"%H+" W:%d"%W+" C_IN:%d"%C_IN
                LUT[device][name]=latency
        elif op=='Zero':
            Hset=[64,32]
            Wset=[64,32]
            Cset=[48,96]
            STRIDEset=[1,2]
            for i in range(len(Hset)):
                H=Hset[i]
                W=Wset[i]
                for C in Cset:
                    for STRIDE in STRIDEset:
                        latency = Zero._latency(H,W,C,C,STRIDE)
                        name=op+" H:%d"%H+" W:%d"%W+" C_IN:%d"%C+" C_OUT:%d"%C+" STRIDE:%d"%STRIDE
                        LUT[device][name]=latency
        elif op=='ReLUConvBN':
            Scale=[64,32]
            C_INset=[384,144,192]
            C_OUTset=[48,96]
            KERNELSIZEset=[1]
            STRIDEset=[1]
            PADDINGset=[0]
            AFFINEset=[True]
            for S in Scale:
                for C_IN in C_INset:
                    for C_OUT in C_OUTset:
                        if C_IN>C_OUT:
                            for KERNELSIZE in KERNELSIZEset:
                                for STRIDE in STRIDEset:
                                    for PADDING in PADDINGset:
                                        for AFFINE in AFFINEset:
                                            latency = ReLUConvBN._latency(S,S,C_IN,C_OUT,KERNELSIZE,STRIDE,PADDING,AFFINE)
                                            name=op+" H:%d"%S+" W:%d"%S+" C_IN:%d"%C_IN+" C_OUT:%d"%C_OUT+" KERNELSIZE:%d"%KERNELSIZE+" STRIDE:%d"%STRIDE+" PADDING:%d"%PADDING+" AFFINE:%s"%str(AFFINE)
                                            LUT[device][name]=latency
        elif op=='DilConv':
            Scale=[64,32]
            Cset=[48,96]
            KERNELSIZEset=[3,5]
            STRIDEset=[1,2]
            PADDINGset=[2,4]
            DILATIONset=[2]
            AFFINEset=[False]
            for S in Scale:
                for C in Cset:
                    for STRIDE in STRIDEset:
                        for i in range(len(KERNELSIZEset)):
                            KERNELSIZE=KERNELSIZEset[i]
                            PADDING=PADDINGset[i]
                            for DILATION in DILATIONset:
                                for AFFINE in AFFINEset:
                                    latency = DilConv._latency(S,S,C,C,KERNELSIZE,STRIDE,PADDING,DILATION,AFFINE)
                                    name = "DilConv H:%d W:%d C_IN:%d C_OUT:%d KERNELSIZE:%d STRIDE:%d PADDING:%d DILATION:%d AFFINE:%s"%(S,S,C,C,KERNELSIZE,STRIDE,PADDING,DILATION,str(AFFINE))
                                    LUT[device][name]=latency
        elif op=='SepConv':
            Scale=[64,32]
            Cset=[48,96]
            KERNELSIZEset=[3,5]
            STRIDEset=[1,2]
            PADDINGset=[1,2]
            AFFINEset=[False]
            for S in Scale:
                for C in Cset:
                    for STRIDE in STRIDEset:
                        for i in range(len(KERNELSIZEset)):
                            KERNELSIZE=KERNELSIZEset[i]
                            PADDING=PADDINGset[i]
                            for AFFINE in AFFINEset:
                                latency = SepConv._latency(S,S,C,C,KERNELSIZE,STRIDE,PADDING,AFFINE)
                                name = "SepConv H:%d W:%d C_IN:%d C_OUT:%d KERNELSIZE:%d STRIDE:%d PADDING:%d AFFINE:%s"%(S,S,C,C,KERNELSIZE,STRIDE,PADDING,str(AFFINE))
                                LUT[device][name]=latency
        elif op=='FactorizedReduce':
            Hset=[64]
            Wset=[64]
            C_INset=[192,96]
            C_OUTset=[96]
            AFFINEset=[False]
            for i in range(len(Hset)):
                H=Hset[i]
                W=Wset[i]
                for C_IN in C_INset:
                    for C_OUT in C_OUTset:
                        if C_IN>=C_OUT:
                            for AFFINE in AFFINEset:
                                latency = FactorizedReduce._latency(H,W,C_IN,C_OUT,AFFINE)
                                name="FactorizedReduce H:%d W:%d C_IN:%d C_OUT:%d AFFINE:%s"%(H,W,C_IN,C_OUT,AFFINE)
                                LUT[device][name]=latency
    print(LUT)
    with open(filename, 'wb') as f:
        pickle.dump(LUT, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
    