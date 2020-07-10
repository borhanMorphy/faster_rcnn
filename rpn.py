import torch.nn as nn
from typing import Dict
import torch
import torch.nn.functional as F

class DetectionLayer(nn.Module):
    pass

class RPN(nn.Module):
    """Input features represents vgg16 backbone and n=3 is set because of the `faster rcnn` parper

    Number of anchors selected from paper, where num_anchors: num_scales * num_ratios
    #! (Change number anchors if scales or ratios are changed)

    """
    def __init__(self, features:int=512, n:int=3, num_anchors:int=6):
        assert n % 2 == 1,"kernel size must be odd"

        # padding calculation for same input output dimensions
        padding = int((n-1) / 2)

        self.base_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features,
                kernel_size=n, stride=1, padding=padding),
            nn.ReLU(inplace=True))

        self.cls_conv_layer = nn.Conv2d(
            in_channels=features, out_channels=num_anchors*2,
            kernel_size=1, stride=0, padding=0)

        self.reg_conv_layer = nn.Conv2d(
            in_channels=features, out_channels=num_anchors*4,
            kernel_size=1, stride=0, padding=0)

        self.detection_layer = None # TODO


    def forward(self, fmap:torch.Tensor, targets:Dict=None):
        pass