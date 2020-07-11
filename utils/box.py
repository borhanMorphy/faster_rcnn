import torch
from typing import List,Tuple

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

def apply_box_regressions(default_boxes:torch.Tensor, regs:torch.Tensor) -> torch.Tensor:
    """
        default_boxes: 
        regs: bs x N x 4

        returns:
            boxes: bs x (num_anchors * fmap_h * fmap_w) x 4 as xmin ymin xmax ymax
    """
    # num_anchors x fmap_h * fmap_w x 4 => N x 4
    # convert xyxy => cxcywh
    anchor_boxes = xyxy2cxcywh(default_boxes.reshape(-1,4))

    cx = anchor_boxes[:, 2] * regs[:,:,0] + anchor_boxes[:, 0]
    cy = anchor_boxes[:, 3] * regs[:,:,1] + anchor_boxes[:, 1]
    w = torch.exp(regs[:,:,2]) * anchor_boxes[:, 2]
    h = torch.exp(regs[:,:,3]) * anchor_boxes[:, 3]

    return cxcywh2xyxy(torch.stack([cx,cy,w,h],dim=-1))


def get_box_regressions():
    pass

def cxcywh2xyxy(o_boxes:torch.Tensor):
    boxes = o_boxes.clone()
    single_batch = len(boxes.shape) == 2
    boxes = boxes.unsqueeze(0) if single_batch else boxes # N,4 => 1,N,4

    w_half = boxes[:, :, 2] / 2
    h_half = boxes[:, :, 3] / 2

    boxes[:, :, 2] = boxes[:, :, 0] + w_half
    boxes[:, :, 3] = boxes[:, :, 1] + h_half
    boxes[:, :, 0] = boxes[:, :, 0] - w_half
    boxes[:, :, 1] = boxes[:, :, 1] - h_half

    return boxes.squeeze(0) if single_batch else boxes

def xyxy2cxcywh(o_boxes:torch.Tensor):
    boxes = o_boxes.clone()
    single_batch = len(boxes.shape) == 2
    boxes = boxes.unsqueeze(0) if single_batch else boxes # N,4 => 1,N,4

    # x1,y1,x2,y2
    w = boxes[:, :, 2] - boxes[:, :, 0]
    h = boxes[:, :, 3] - boxes[:, :, 1]

    boxes[:, :, :2] = (boxes[:, :, :2] + boxes[:, :, 2:]) / 2
    boxes[:, :, 2] = w
    boxes[:, :, 3] = h

    return boxes.squeeze(0) if single_batch else boxes

def jaccard_vectorized(box_a:torch.Tensor, box_b:torch.Tensor) -> torch.Tensor:
    inter = intersect(box_a,box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
                (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter) # [A,B]

    union = area_a + area_b - inter
    return inter / union

def intersect(box_a:torch.Tensor, box_b:torch.Tensor) -> torch.Tensor:
    """
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]

    Args:
        box_a (torch.Tensor): [description]
        box_b (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def clip_boxes(c_boxes:torch.Tensor, img_dims:Tuple):
    """Clips boxes that exceeds image region

    Params:
        boxes: bs x N x 4 as xmin ymin xmax ymax
        img_dims: image height, image width
    """
    h,w = img_dims
    boxes = c_boxes.clone()

    boxes[..., :2] = torch.clamp(boxes[..., :2], min=0)
    boxes[..., 2] = torch.clamp(boxes[..., 2], max=w)
    boxes[..., 3] = torch.clamp(boxes[..., 3], max=h)

    return boxes




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