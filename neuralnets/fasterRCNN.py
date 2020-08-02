import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool,RoIPool
from typing import List,Dict,Tuple
from .fastRCNN import FastRCNNHead
from .rpn import RPNHead
from collections import OrderedDict

class FasterRCNN(nn.Module):
    def __init__(self, backbone:nn.Module, num_classes:int):
        super(FasterRCNN,self).__init__()
        self.backbone = backbone
        self.rpn = RPNHead(backbone.output_channels)
        self.head = FastRCNNHead(backbone.output_channels, num_classes)

    def forward(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]=None):
        img_dims = batch.shape[-2:]

        fmaps = self.backbone(batch)

        batched_rois = self.rpn(fmaps, img_dims, targets=targets)

        if targets is not None:
            batched_rois,rpn_losses = batched_rois

        fmaps = OrderedDict([('0',fmaps)])
        img_dims = [img_dims]

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

    def training_step(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
        batched_rois,batched_dets,losses = self.forward(batch, targets=targets)

        joint_loss = losses['rpn_cls_loss'] + losses['rpn_reg_loss'] +\
            losses['head_cls_loss'] + losses['head_reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()
        losses['loss'] = joint_loss
        return losses

    @torch.no_grad()
    def validation_step(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
        batched_rois,batched_dets,losses = self.forward(batch, targets=targets)

        det_targets = []
        for target in targets:
            gt_boxes = target['boxes']
            labels = target['labels'].to(gt_boxes.device, gt_boxes.dtype).unsqueeze(-1)
            det_targets.append(torch.cat([gt_boxes,labels], dim=-1).cpu())

        roi_targets = [det_target[:,:4] for det_target in det_targets] # K,5 => K,4
        
        losses['loss'] = losses['rpn_cls_loss'] + losses['rpn_reg_loss'] +\
            losses['head_cls_loss'] + losses['head_reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()

        detections = {
            'rpn':{
                'predictions': [rois.cpu() for rois in batched_rois],
                'ground_truths': roi_targets.copy()
            },
            'head':{
                'predictions': [dets.cpu() for dets in batched_dets],
                'ground_truths': det_targets.copy()
            }
        }
        
        return detections,losses