import torch
from typing import List

def generate_default_boxes(anchors, fmap_dims,
        effective_stride:str, dtype=torch.float32, device:str='cpu'):
    h,w = fmap_dims
    grid_x = torch.arange(w, dtype=torch.float32, device=device) * effective_stride
    grid_y = torch.arange(h, dtype=torch.float32, device=device) * effective_stride
    grid_x,grid_y = torch.meshgrid(grid_x,grid_y)
    # h*w,2
    grids = torch.cat([grid_y.reshape(-1,1),grid_x.reshape(-1,1)], dim=1)
    dboxes = anchors.repeat(h*w,1,1).permute(1,0,2)
    dboxes[:,:,:2] += grids
    dboxes[:,:,2:] += grids
    return dboxes

def generate_anchors(effective_stride:int, ratios:List=[0.5,1,2],
        scales:List=[0.5,1,2], dtype=torch.float32, device:str='cpu'):

    r = torch.tensor([ratios], dtype=dtype, device=device) # (len(ratios),1)
    s = torch.tensor([scales], dtype=dtype, device=device) * effective_stride # (len(scales),)
    anchors = torch.tensor([[1,1,effective_stride,effective_stride]], dtype=dtype, device=device) - 1 # (1,4)

    # generate anchors using given ratios
    x_ctr,y_ctr,w,h = _anchor2vector(anchors)
    w = torch.round( torch.sqrt((w*h)/r) )
    h = torch.round( w*r )
    anchors = _vector2anchor(x_ctr,y_ctr,w,h)

    # generate anchors using given scales
    x_ctr,y_ctr,w,h = _anchor2vector(anchors)

    w = w.unsqueeze(1) * s # [1,3] . [3,3]
    h = h.unsqueeze(1) * s

    # TODO handle `+1`
    return _vector2anchor(x_ctr.repeat(3), y_ctr.repeat(3), w.reshape(1,-1), h.reshape(1,-1)) + 1

def _vector2anchor(x_ctr:torch.Tensor, y_ctr:torch.Tensor, w:torch.Tensor, h:torch.Tensor):
    xmin = x_ctr - 0.5*(w-1)
    ymin = y_ctr - 0.5*(h-1)
    xmax = x_ctr + 0.5*(w-1)
    ymax = y_ctr + 0.5*(h-1)

    return torch.cat([xmin,ymin,xmax,ymax], dim=0).T

def _anchor2vector(anchors:torch.Tensor):
    w = anchors[:, 2] - anchors[:, 0] + 1
    h = anchors[:, 3] - anchors[:, 1] + 1
    x_ctr = anchors[:, 0] + 0.5 * (w-1)
    y_ctr = anchors[:, 1] + 0.5 * (h-1)
    return x_ctr,y_ctr,w,h

if __name__ == "__main__":
    from cv2 import cv2
    import numpy as np
    import torchvision.models as models

    img = cv2.imread("/home/borhan-morphy/Desktop/temp_images/lpr_test.jpg")
    data = torch.from_numpy(img.astype(np.float32) / 255).permute(2,0,1).unsqueeze(0)
    m = models.vgg16(pretrained=True)
    m.eval()

    with torch.no_grad():
        fmap = m.features[:-1](data)

    effective_stride = 16
    anchors = generate_anchors(effective_stride)
    b,c,h,w = fmap.shape
    anchors = generate_default_boxes(anchors,(h,w),effective_stride)

    for boxes in anchors:
        c_img = img.copy()
        for x1,y1,x2,y2 in boxes.long().numpy():
            d_img = cv2.rectangle(c_img,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.imshow("",d_img)
            cv2.waitKey(0)