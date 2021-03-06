import torch
import torch.nn as nn
from typing import List,Tuple,Dict

class AnchorGenerator(nn.Module):
    def __init__(self, sizes:List[int], aspect_ratios:List[float]):
        super(AnchorGenerator,self).__init__()

        self.num_anchors = len(sizes) * len(aspect_ratios)
        # generate anchors centered at (0,0)
        self.base_anchors = torch.cat([ # nA x 4 anchors centered at (0,0) point
            self.generate_base_anchors([size],aspect_ratios) 
            for size in sizes],dim=0)

    @staticmethod
    def generate_base_anchors(sizes:List[int], aspect_ratios:List[float]):
        # *HINT: do not give all sizes at once, give one by one in order to get correct order of anchors
        sizes = torch.tensor(sizes, dtype=torch.float32)
        aspect_ratios = torch.tensor(aspect_ratios, dtype=torch.float32)
        
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios.unsqueeze(0).t() * sizes.unsqueeze(0)).view(-1)
        hs = (h_ratios.unsqueeze(0).t() * sizes.unsqueeze(0)).view(-1)

        return (torch.stack([-ws, -hs, ws, hs], dim=1) / 2).round()

    def forward(self, fmap_dims:Tuple[int,int], img_dims:Tuple[int,int],
            dtype=torch.float32, device='cpu') -> torch.Tensor:
        """
        Params:
            fmap_dims Tuple[int,int]: h',w'
            img_dims Tuple[int,int]: h,w
        
        Returns:
            anchors torch.Tensor((nA*h'*w'),4) as xmin,ymin,xmax,ymax
        """
        # fmap_dims: h',w'
        # img_dims: h,w
        fh,fw = fmap_dims
        h,w = img_dims

        shift_x = torch.arange(0,fw, dtype=dtype, device=device) * int(w/fw)
        shift_y = torch.arange(0,fh, dtype=dtype, device=device) * int(h/fh)
        shift_y,shift_x = torch.meshgrid(shift_y,shift_x)

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        if self.base_anchors.dtype != dtype:
            self.base_anchors = self.base_anchors.to(dtype=dtype)
        if self.base_anchors.device != device:
            self.base_anchors = self.base_anchors.to(device=device)

        return (shifts.view(-1, 1, 4) + self.base_anchors.view(1, -1, 4)).reshape(-1, 4)

def offsets2boxes(offsets:torch.Tensor, anchors:torch.Tensor):
    """
    Params:
        offsets torch.Tensor(bs,N,4): as dx,dy,dw,dh
        anchors torch.Tensor(N,4): as xmin,ymin,xmax,ymax

    Returns:
        boxes torch.Tensor(N,4): as xmin,ymin,xmax,ymax
    """
    assert len(offsets.shape) == 3
    bs = offsets.size(0)
    deltas = offsets.clone()

    deltas = deltas.reshape(-1,4)

    c_anchors = anchors.to(deltas.device).repeat(bs,1,1).reshape(-1,4)
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    wa = c_anchors[:, 2::4] - c_anchors[:, 0::4]
    ha = c_anchors[:, 3::4] - c_anchors[:, 1::4]
    cxa = c_anchors[:, 0::4] + .5 * wa
    cya = c_anchors[:, 1::4] + .5 * ha

    cx = dx * wa + cxa
    cy = dy * ha + cya
    w = torch.exp(dw) * wa
    h = torch.exp(dh) * ha

    xmin = cx - .5 * w
    ymin = cy - .5 * h
    xmax = cx + .5 * w
    ymax = cy + .5 * h

    return torch.cat([xmin,ymin,xmax,ymax], dim=-1).reshape(bs,-1,4)

def boxes2offsets(boxes:torch.Tensor, anchors:torch.Tensor):
    """
    Params:
        boxes torch.Tensor(N,4): as xmin,ymin,xmax,ymax
        anchors torch.Tensor(N,4): as xmin,ymin,xmax,ymax

    Returns:
        deltas torch.Tensor(N,4): as dx,dy,dw,dh
    """
    eps = 1e-16
    anchors = anchors.to(boxes.device)
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + .5 * wa
    cya = anchors[:, 1] + .5 * ha

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + .5 * w
    cy = boxes[:, 1] + .5 * h

    dx = (cx - cxa) / wa
    dy = (cy - cya) / ha
    dw = torch.log(w / wa + eps)
    dh = torch.log(h / ha + eps)

    return torch.stack([dx,dy,dw,dh], dim=-1)

if __name__ == "__main__":
    anchor_generator = AnchorGenerator([128,256,512],[0.5,1,2])

    fmap_dims = [(96,64),(44,50)]
    img_dims = [(960,640),(600,700)]

    batch_anchors = anchor_generator(fmap_dims,img_dims)

    for anchors in batch_anchors:
        print(anchors.shape)