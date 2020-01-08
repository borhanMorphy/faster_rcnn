import math
import argparse

def get_arguments():
    parser = argparse.ArgumentParser("feature map dimention calculator")
    parser.add_argument("--width","-wt",required=True,type=int)
    parser.add_argument("--height","-ht",required=True,type=int)
    parser.add_argument("--model","-m",required=True,type=str,choices=["vgg16","alexnet"])
    return parser.parse_args()

class Op:
    def __init__(self,kernel,stride,padding,featuremap):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.featuremap = featuremap

    def __call__(self,w,h,c):
        return self._calc(w),self._calc(h),self.featuremap

    def _calc(self,d):
        return math.floor((d+2*self.padding-self.kernel)/self.stride+1)

class Conv(Op):
    pass

class Pool(Op):
    def __init__(self,kernel,stride,padding):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def __call__(self,w,h,c):
        return self._calc(w),self._calc(h),c

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

def inference(model,w,h,c):
    for layer in model:
        w,h,c = layer(w,h,c)
    return w,h,c

if __name__ == '__main__':
    args = get_arguments()
    w,h = args.width,args.height
    model_name = args.model
    w,h,c = int(w),int(h),3
    if model_name == "vgg16":
        model = vgg16
    elif model_name == "alexnet":
        model = alexnet
    else:
        raise ValueError("model not defined")

    ws,hs,cs = inference(model,w,h,3)
    print("order: height x width x channel")
    #print(f"model architecture:\n{model}")
    print(f"model name: {model_name}")
    print(f"original input dimentions: {h}x{w}x{c}")
    print(f"output dimentions: {hs}x{ws}x{cs}")