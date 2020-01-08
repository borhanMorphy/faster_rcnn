import math

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
    import sys
    w,h = sys.argv[-2:]
    w,h = int(w),int(h)
    ws,hs,cs = inference(alexnet,w,h,3)
    print(w/ws)
    print(h/hs)