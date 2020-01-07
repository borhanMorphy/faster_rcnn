
class Conv():
    def __init__(self,kernel,stride,padding,fc):
        self.stride = stride
        self.padding = padding
        self.kernel = kernel
        self.fc = fc
    
    def __call__(self,w,h,c):
        return self._calc(w),self._calc(h),self.fc
    
    def _calc(self,d):
        return int((d-self.kernel+2*self.padding)/self.stride + 1)

class Pool():
    def __init__(self,kernel,stride,padding):
        self.stride = stride
        self.padding = padding
        self.kernel = kernel
    
    def __call__(self,w,h,c):
        return self._calc(w),self._calc(h),c
    
    def _calc(self,d):
        return int((d-self.kernel+2*self.padding)/self.stride + 1)

def alexnet(w,h,c):
    ops = []
    ops.append(Conv(11,4,0,96))
    ops.append(Pool(3,2,0))

    ops.append(Conv(5,1,2,256))
    ops.append(Pool(3,2,0))
    
    ops.append(Conv(3,1,1,384))
    ops.append(Conv(3,1,1,384))
    ops.append(Conv(3,1,1,256))
    ops.append(Pool(3,2,0))

    for op in ops:
        w,h,c = op(w,h,c)
    
    return w,h,c

def vgg16(w,h,c):    
    ops = []
    ops.append(Conv(3,1,1,64))
    ops.append(Conv(3,1,1,64))
    ops.append(Pool(2,2,0))

    ops.append(Conv(3,1,1,128))
    ops.append(Conv(3,1,1,128))
    ops.append(Pool(2,2,0))

    ops.append(Conv(3,1,1,256))
    ops.append(Conv(3,1,1,256))
    ops.append(Conv(3,1,1,256))
    ops.append(Pool(2,2,0))

    ops.append(Conv(3,1,1,512))
    ops.append(Conv(3,1,1,512))
    ops.append(Conv(3,1,1,512))
    ops.append(Pool(2,2,0))

    ops.append(Conv(3,1,1,512))
    ops.append(Conv(3,1,1,512))
    ops.append(Conv(3,1,1,512))
    ops.append(Pool(2,2,0))

    for op in ops:
        w,h,c = op(w,h,c)
    
    return w,h,c

if __name__ == '__main__':
    import sys
    w,h = sys.argv[-2:]
    w = int(w)
    h = int(h)
    ws,hs,cs = alexnet(w,h,3)
    print(ws,hs,cs)

    print(h/hs)
    print(w/ws)
