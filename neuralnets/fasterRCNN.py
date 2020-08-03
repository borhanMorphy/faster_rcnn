import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool,RoIPool
from typing import List,Dict,Tuple
from .fastRCNN import FastRCNNMultiHead,FastRCNNSingleHead
from .rpn import RPNHead
from collections import OrderedDict

class FasterRCNN(nn.Module):
    def __init__(self, backbone:nn.Module, num_classes:int,
            # RPN
            rpn_kernel_size:int=3, rpn_anchor_sizes:List[int]=[128,256,512],
            rpn_anchor_aspect_ratios:List[float]=[0.5,1,2],
            rpn_nms_threshold:float=0.7, rpn_train_keep_pre_nms:int=2000,
            rpn_train_keep_post_nms:int=2000,
            rpn_test_keep_pre_nms:int=1000, rpn_test_keep_post_nms:int=300,
            rpn_batch_size_per_image:int=256, rpn_batch_positive_ratio:float=0.5,
            rpn_fg_iou_threshold:float=0.7, rpn_bg_iou_threshold:float=0.3, rpn_anchor_generator=None,
            # HEAD
            roi_output_size:int=7, head_hidden_channels:int=1024,
            head_batch_size_per_image:int=512, head_batch_positive_ratio:float=.25,
            head_fg_iou_threshold:float=0.5, head_bg_iou_threshold:float=0.5,
            head_conf_threshold:float=0.05, head_nms_threshold:float=0.5, head_keep_top_n:int=100):

        super(FasterRCNN,self).__init__()
        self.backbone = backbone
        self.rpn = RPNHead(backbone.output_channels, rpn_kernel_size=rpn_kernel_size,
            anchor_sizes=rpn_anchor_sizes, anchor_aspect_ratios=rpn_anchor_aspect_ratios,
            nms_threshold=rpn_nms_threshold,
            train_keep_pre_nms=rpn_train_keep_pre_nms, train_keep_post_nms=rpn_train_keep_post_nms,
            test_keep_pre_nms=rpn_test_keep_pre_nms, test_keep_post_nms=rpn_test_keep_post_nms,
            batch_size_per_image=rpn_batch_size_per_image, batch_positive_ratio=rpn_batch_positive_ratio,
            fg_iou_threshold=rpn_fg_iou_threshold, bg_iou_threshold=rpn_bg_iou_threshold,
            anchor_generator=rpn_anchor_generator)

        if num_classes > 2:
            self.head = FastRCNNMultiHead(backbone.output_channels, num_classes,
                roi_output_size=roi_output_size, hidden_channels=head_hidden_channels,
                batch_size_per_image=head_batch_size_per_image, batch_positive_ratio=head_batch_positive_ratio,
                fg_iou_threshold=head_fg_iou_threshold, bg_iou_threshold=head_bg_iou_threshold,
                conf_threshold=head_conf_threshold, nms_threshold=head_nms_threshold, keep_top_n=head_keep_top_n)
        else:
            self.head = FastRCNNSingleHead(backbone.output_channels,
                roi_output_size=roi_output_size, hidden_channels=head_hidden_channels,
                batch_size_per_image=head_batch_size_per_image, batch_positive_ratio=head_batch_positive_ratio,
                fg_iou_threshold=head_fg_iou_threshold, bg_iou_threshold=head_bg_iou_threshold,
                conf_threshold=head_conf_threshold, nms_threshold=head_nms_threshold, keep_top_n=head_keep_top_n)

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

    def validation_step(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
        with torch.no_grad():
            batched_rois,batched_dets,losses = self.forward(batch, targets=targets)

        det_targets = []
        for target in targets:
            gt_boxes = target['boxes'].cpu()
            labels = target['labels'].to(gt_boxes.device, gt_boxes.dtype).unsqueeze(-1)
            det_targets.append(torch.cat([gt_boxes,labels], dim=-1))

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