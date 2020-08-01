import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool,RoIPool
from typing import List,Dict,Tuple
from .fastRCNN import FastRCNNHead
from .rpn import RPNHead

class FasterRCNN(nn.Module):
    def __init__(self, backbone:nn.Module, num_classes:int):
        super(FasterRCNN,self).__init__()
        self.backbone = backbone
        self.rpn = RPNHead(backbone.output_channels)
        self.head = FastRCNNHead(backbone.output_channels, num_classes)

    def forward(self, images:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]=None):
        img_dims = [img.shape[-2:] for img in images]

        fmaps = [self.backbone(img) for img in images]

        batched_rois = self.rpn(fmaps, img_dims, targets=targets)

        if targets is not None:
            batched_rois,rpn_losses = batched_rois

        batched_dets = self.head(fmaps,[rois[:,:4] for rois in batched_rois], img_dims, targets=targets)

        if targets is not None:
            batched_dets,head_losses = batched_dets
            losses = {
                'rpn_cls_loss': rpn_losses['cls_loss'],
                'rpn_reg_loss': rpn_losses['reg_loss'],
                'head_cls_loss': head_losses['cls_loss'],
                'head_reg_loss': head_losses['reg_loss']
            }
            return batched_rois,batched_dets,losses

        return batched_dets

    def training_step(self, batch:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]):
        batched_rois,batched_dets,losses = self.forward(batch, targets=targets)

        joint_loss = losses['rpn_cls_loss'] + losses['rpn_reg_loss'] +\
            losses['head_cls_loss'] + losses['head_reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()
        losses['loss'] = joint_loss
        return losses

    @torch.no_grad()
    def validation_step(self, batch:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]):
        batched_rois,batched_dets,losses = self.forward(batch, targets=targets)

        det_targets = [target['boxes'].cpu() for target in targets] # K,4
        losses['loss'] = losses['rpn_cls_loss'] + losses['rpn_reg_loss'] +\
            losses['head_cls_loss'] + losses['head_reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()

        detections = {
            'rpn':{
                'predictions': [rois.cpu() for rois in batched_rois],
                'ground_truths': det_targets.copy()
            },
            'head':{
                'predictions': [dets[:,:5].cpu() for dets in batched_dets],
                'ground_truths': det_targets.copy()
            }
        }
        
        return detections,losses