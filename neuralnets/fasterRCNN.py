import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool,RoIPool
from typing import List,Dict,Tuple
from .fastRCNN import FastRCNN
from .rpn import RPN

class FasterRCNN(nn.Module):
    def __init__(self, num_classes:int,
            backbone:nn.Module, features:int, effective_stride:int,
            roi_output_size:int=7, head_hidden_channels:int=1024,
            rpn_kernel_size:int=3, anchor_scales:List=[0.5,1,2],
            anchor_ratios:List=[0.5,1,2]):
        super(FasterRCNN,self).__init__()
        self.backbone = backbone

        self.rpn = RPN(features=features, n=rpn_kernel_size,
            effective_stride=effective_stride, anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios) # TODO add param options

        self.head = FastRCNN(num_classes,effective_stride,
            output_size=roi_output_size, features=features,
            hidden_channels=head_hidden_channels)

    def eval(self, *args, **kwargs):
        self.requires_grad_(False)
        return super().eval(*args,**kwargs)

    def train(self, *args, **kwargs):
        self.requires_grad_(True)
        return super().train(*args,**kwargs)

    def forward(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]=None):
        _,_,ih,iw = batch.shape

        fmap = self.backbone(batch)

        rois = self.rpn(fmap, (ih,iw), targets=targets)

        if targets is not None:
            rois,rpn_losses = rois

        dets = self.head(fmap,rois,targets=targets)

        if targets is not None:
            dets,head_losses = dets
            losses = {
                'rpn_cls_loss':rpn_losses['cls_loss'],
                'rpn_reg_loss':rpn_losses['regloss'],
                'head_cls_loss':head_losses['cls_loss'],
                'head_reg_loss':head_losses['reg_loss']
            }
            return dets,losses

        return dets


    def training_step(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
        dets,losses = self.forward(batch,targets=targets)
