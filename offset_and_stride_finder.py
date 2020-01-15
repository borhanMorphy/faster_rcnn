import math
import argparse
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser("base stride and offset calculator")
    parser.add_argument("--model","-m",required=True,type=str,choices=["vgg16","alexnet","rnet"])
    return parser.parse_args()

class Op:
    def __init__(self,kernel,stride,padding,featuremap):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.featuremap = featuremap

    def __call__(self,w,h,c,reverse=False):
        if not reverse:
            return self._calc(w),self._calc(h),self.featuremap
        else:
            return self._rev(w),self._rev(h)

    def _calc(self,d):
        return math.floor((d+2*self.padding-self.kernel)/self.stride+1)

    def _rev(self,d):
        return self.stride*(d-1)+self.kernel-2*self.padding

class Conv(Op):
    pass

class Pool(Op):
    def __init__(self,kernel,stride,padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def __call__(self,w,h,c,reverse=False):
        if not reverse:
            return self._calc(w),self._calc(h),c
        else:
            return super().__call__(w,h,c,reverse)

alexnet = [
    Conv(11,4,0,96),
    Pool(3,2,0),

    Conv(5,1,2,256),
    Pool(3,2,0),

    Conv(3,1,1,384),
    
    Conv(3,1,1,384),

    Conv(3,1,1,256)
]

vgg16 = [
    Conv(3,1,1,64),
    Conv(3,1,1,64),
    Pool(2,2,0),

    Conv(3,1,1,128),
    Conv(3,1,1,128),
    Pool(2,2,0),

    Conv(3,1,1,256),
    Conv(3,1,1,256),
    Conv(3,1,1,256),
    Pool(2,2,0),

    Conv(3,1,1,512),
    Conv(3,1,1,512),
    Conv(3,1,1,512),
    Pool(2,2,0),

    Conv(3,1,1,512),
    Conv(3,1,1,512),
    Conv(3,1,1,512)
]

rnet = [
    Conv(3,1,0,10),
    Pool(2,2,0),
    Conv(3,1,0,16),
    Conv(3,1,0,32),
    Conv(1,1,0,2)
]

def inference(model,w,h,c):
    for layer in model:
        w,h,c = layer(w,h,c)
    return w,h,c

def reverse_inference(model,w,h):
    for layer in reversed(model):
        w,h = layer(w,h,0,reverse=True)
    return w,h

def calc_base_stride(model) -> int:
    s = [0,0]
    for i in range(2):
        d = i+1
        for layer in reversed(model):
            d,_ = layer(d,d,0,reverse=True)
        s[i] = d
    return s[1]-s[0]

def calc_base_offset(model) -> float:
    d = 1
    for layer in reversed(model):
        d,_ = layer(d,d,0,reverse=True)
    return d/2

if __name__ == '__main__':
    
    args = get_arguments()
    
    model_name = args.model
    
    if model_name == "vgg16":
        model = vgg16
    elif model_name == "alexnet":
        model = alexnet
    elif model_name == "rnet":
        model = rnet
    else:
        raise ValueError("model not defined")

    offset = calc_base_offset(model)
    stride = calc_base_stride(model)
    
    print(f"model: {model_name}\nbase stride: {stride}\nbase offset: {offset}")