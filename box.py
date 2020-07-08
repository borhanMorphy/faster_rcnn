import torch
from typing import List
import numpy as np

def generate_anchors(effective_stride:int, ratios:List=[0.5,1,2],
        scales:List=[0.5,1,2], dtype=torch.float32, device:str='cpu'):

    r = torch.tensor(ratios, dtype=dtype, device=device)
    s = torch.tensor(scales, dtype=dtype, device=device) * effective_stride
    window = torch.tensor([0,0,effective_stride-1,effective_stride-1], dtype=dtype, device=device)

    area = effective_stride*effective_stride

    area_r = area / r

    centers = (window[2:] * 0.5).repeat(len(ratios)).reshape(-1,2)

    ws = torch.round(torch.sqrt(area_r))
    hs = torch.round(ws * r)
    wh = torch.stack([ws,hs], dim=0).T - 1

    anchors = torch.cat([centers,wh], dim=-1) # cx,cy,w,h

    # converting cx,cy,w,h => xmin,ymin,xmax,ymax
    anchors[:,2:] = anchors[:,:2] + wh/2
    anchors[:,:2] -= wh/2

    new_anchors = []
    for i in range(anchors.size(0)):
        new_anchors.append( scale_anchor(anchors[i], s) )

    return anchors

def anchorgrid2vector(anchor):
    # xmin,ymin,xmax,ymax grid
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    cx = anchor[0] + 0.5 * (w - 1)
    cy = anchor[1] + 0.5 * (h - 1)
    return cx, cy, w, h

def vector2anchorgrid(box):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def scale_anchor(anchor:torch.Tensor, scales:torch.Tensor):
    #anchor : xmin,ymin,xmax,ymax
    cx = (anchor[0] + anchor[2]) / 2
    cy = (anchor[1] + anchor[3]) / 2
    centers = torch.stack([cx,cy],dim=0).repeat(scales.size(0)).reshape(-1,2)

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1

    ws = w*scales
    hs = h*scales
    wh = torch.stack([ws,hs], dim=0).T

    anchors = torch.cat([centers,wh], dim=-1) # cx,cy,w,h

    # converting cx,cy,w,h => xmin,ymin,xmax,ymax
    anchors[:,2:] = anchors[:,:2] + wh/2
    anchors[:,:2] -= wh/2

    print(anchors);exit(0)

    return anchors


def generate_anchors_original(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == "__main__":
    #print(generate_anchors(16))
    print(generate_anchors_original(base_size=16))