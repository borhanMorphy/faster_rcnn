import math
import argparse
import numpy as np
import cv2

def get_arguments():
    parser = argparse.ArgumentParser("feature map dimention calculator")
    parser.add_argument("--input","-i",required=True,type=str)
    parser.add_argument("--model","-m",required=True,type=str,choices=["vgg16","alexnet","pnet"])
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
pnet = [
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

def calc_point_count(model,w,h,c) -> tuple:
    base_offset = calc_base_offset(model)
    base_stride = calc_base_stride(model)
    assert min(w,h) >= base_offset*2,f"dimentions must be greater or equal to than {base_offset*2}"
    offset_x = ((w-base_offset*2)%base_stride)/2 + base_offset
    offset_y = ((h-base_offset*2)%base_stride)/2 + base_offset
    
    i = ((w-2*offset_x) // base_stride) + 1
    j = ((h-2*offset_y) // base_stride) + 1
    return (int(i),int(j)),(offset_x,offset_y),base_stride

def cal_point_center_coords(model,w,h,c) -> tuple:
    points,offsets,stride = calc_point_count(model,w,h,c)
    i,j = points
    offset_x,offset_y = offsets
    centers = []
    
    for ci in range(i):
        cx = ci*stride+offset_x
        for cj in range(j):
            cy = cj*stride+offset_y
            centers.append([cx,cy]) 
    return centers,int(i*j),stride

if __name__ == '__main__':
    
    args = get_arguments()
    img = cv2.imread(args.input)
    img = cv2.resize(img,(28,28))
    model_name = args.model
    h,w,c = img.shape
    if model_name == "vgg16":
        model = vgg16
    elif model_name == "alexnet":
        model = alexnet
    elif model_name == "pnet":
        model = pnet
    else:
        raise ValueError("model not defined")

    ws,hs,cs = inference(model,w,h,3)
    print("order: height x width x channel")
    #print(f"model architecture:\n{model}")
    print(f"model name: {model_name}")
    print(f"original input dimentions: {h}x{w}x{c}")
    print(f"output dimentions: {hs}x{ws}x{cs}")
    centers,N,stride = cal_point_center_coords(model,w,h,c)
    print(f"point stride: {stride}\t point count: {N}")
    print("point centers: ",centers)
    print(len(centers))
    for x,y in centers:
        cv2.circle(img,(int(x),int(y)),5,(255,0,0),1)
    cv2.imshow("",img)
    cv2.waitKey(0)