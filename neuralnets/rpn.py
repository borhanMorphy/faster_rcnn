import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,List,Tuple
from torchvision.ops import boxes as box_ops
from cv2 import cv2
import numpy as np
from utils import (
    offsets2boxes,
    boxes2offsets,
    AnchorGenerator,
    build_targets,
    sample_fg_bg
)

class PredictionLayer(nn.Module):
    def __init__(self, num_anchors:int, backbone_features:int, n:int=3):
        super(PredictionLayer,self).__init__()

        assert n % 2 == 1,"kernel size must be odd"

        # padding calculation for same input output dimensions
        padding = int((n-1) / 2)

        self.base_conv_layer = nn.Conv2d(in_channels=backbone_features, out_channels=backbone_features,
            kernel_size=n, stride=1, padding=padding)

        self.cls_conv_layer = nn.Conv2d(
            in_channels=backbone_features, out_channels=num_anchors,
            kernel_size=1, stride=1, padding=0)

        self.reg_conv_layer = nn.Conv2d(
            in_channels=backbone_features, out_channels=num_anchors*4,
            kernel_size=1, stride=1, padding=0)

        # rpn initialization defined in the paper
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    @staticmethod
    def flatten_preds(pred:torch.Tensor, c:int) -> torch.Tensor:
        bs,_,h,w = pred.shape
        pred = pred.view(bs,-1,c,h,w)
        pred = pred.permute(0,3,4,1,2)
        pred = pred.reshape(bs,-1,c)
        return pred

    def forward(self, fmaps:torch.Tensor) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:
        """
        Params:
            fmaps torch.Tensor: feature maps with shape bs x c' x h' x w'

        Returns:
            Tuple[torch.Tensor,torch.Tensor]
                cls_logits: bs x (h'*w'*nA) x 1
                reg_deltas: bs x (h'*w'*nA) x 4 as dx,dy,dw,dh
        """
        outputs = F.relu(self.base_conv_layer(fmaps))
        cls_logits = self.cls_conv_layer(outputs)
        cls_logits = self.flatten_preds(cls_logits, 1)

        reg_deltas = self.reg_conv_layer(outputs)
        reg_deltas = self.flatten_preds(reg_deltas, 4)

        return cls_logits,reg_deltas

class DetectionLayer(nn.Module):
    def __init__(self, anchor_generator):
        super(DetectionLayer,self).__init__()
        self.anchor_generator = anchor_generator
        self.cached_fmap_dims = (-1,-1)

    def forward(self, cls_logits:torch.Tensor, reg_deltas:torch.Tensor,
            fmap_dims:Tuple[int,int], img_dims:Tuple[int,int],
            nms_threshold:float=.7, keep_pre_nms:int=1000, keep_post_nms:int=300,
            dtype=torch.float32, device='cpu'):
        """
        Params:
            cls_logits torch.Tensor: torch.Tensor(bs x (h'*w'*nA) x 1) 
            reg_deltas torch.Tensor: torch.Tensor(bs x (h'*w'*nA) x 4)
            fmap_dims:Tuple[int,int] h',w'
            img_dims:Tuple[int,int] h,w

        Returns:
            batched_dets: List[torch.Tensor(N,5)] as xmin,ymin,xmax,ymax,score
        """
        bs = cls_logits.size(0)
        if self.cached_fmap_dims != fmap_dims:
            # generate anchors for each input
            self.anchors = self.anchor_generator(fmap_dims, img_dims, dtype=dtype, device=device)
            self.cached_fmap_dims = fmap_dims

        batched_dets:List[torch.Tensor] = []

        scores = torch.sigmoid(cls_logits.detach()).reshape(bs,-1)
        offsets = reg_deltas.detach().reshape(bs,-1,4)

        # convert offsets to boxes
        # bs,N,4 | N,4 => bs,N,4 as xmin,ymin,xmax,ymax
        boxes = offsets2boxes(offsets, self.anchors)

        # TODO vectorize this loop
        for i in range(bs):
            single_boxes = boxes[i]
            single_scores = scores[i]
            N = single_scores.size(0)
            
            # select top n
            _,selected_ids = single_scores.topk( min(keep_pre_nms,N) )
            single_scores,single_boxes = single_scores[selected_ids], single_boxes[selected_ids]

            # clip boxes
            single_boxes = box_ops.clip_boxes_to_image(single_boxes, img_dims)

            # nms
            if (single_scores < 0).any():
                print(single_scores[single_scores<0]);exit(0)
            keep = box_ops.nms(single_boxes, single_scores, nms_threshold)
            single_scores,single_boxes = single_scores[keep], single_boxes[keep]

            # post_n
            keep_post_nms = min(keep_post_nms, single_boxes.size(0))
            single_scores,single_boxes = single_scores[:keep_post_nms], single_boxes[:keep_post_nms]

            batched_dets.append( torch.cat([single_boxes,single_scores.unsqueeze(-1)], dim=-1) )

        return batched_dets

class RPNHead(nn.Module):
    def __init__(self, backbone_features:int,
        rpn_kernel_size:int=3,
        anchor_sizes:List[int]=[128,256,512], anchor_aspect_ratios:List[float]=[0.5,1,2],
        nms_threshold:float=0.7,
        train_keep_pre_nms:int=2000, train_keep_post_nms:int=2000,
        test_keep_pre_nms:int=1000, test_keep_post_nms:int=300,
        batch_size_per_image:int=256, batch_positive_ratio:float=0.5,
        fg_iou_threshold:float=0.7, bg_iou_threshold:float=0.3,
        anchor_generator=None):

        super(RPNHead,self).__init__()
        num_anchors = len(anchor_sizes) * len(anchor_aspect_ratios)

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(anchor_sizes, anchor_aspect_ratios)
        else:
            assert anchor_generator.num_anchors == num_anchors,"anchor generator does not match with given parameters"

        self.prediction_layer = PredictionLayer(num_anchors,
            backbone_features, n=rpn_kernel_size)

        self.detection_layer = DetectionLayer(anchor_generator)

        self._params = dict(
            train=tuple((train_keep_pre_nms, train_keep_post_nms)),
            test=tuple((test_keep_pre_nms, test_keep_post_nms)),
            others=tuple((
                batch_size_per_image,batch_positive_ratio,
                fg_iou_threshold,bg_iou_threshold,nms_threshold)))

    def get_params(self):
        if self.training:
            return self._params['train'],self._params['others']
        return self._params['test'],self._params['others']

    def forward(self, fmaps:torch.Tensor, img_dims:Tuple[int,int],
            targets:List[Dict[str,torch.Tensor]]=None):
        (keep_pre_nms, keep_post_nms),\
            (batch_size_per_image, batch_positive_ratio,\
                fg_iou_threshold, bg_iou_threshold,\
                    nms_threshold) = self.get_params()

        dtype = fmaps.dtype
        device = fmaps.device
        bs = fmaps.size(0)

        fmap_dims = fmaps.shape[-2:]

        # cls_logits: (bs x (h'*w'*nA) x 1)
        # reg_deltas: (bs x (h'*w'*nA) x 4) as dx,dy,dw,dh
        cls_logits,reg_deltas = self.prediction_layer(fmaps)

        batched_dets = self.detection_layer(cls_logits, reg_deltas, fmap_dims,
            img_dims, nms_threshold=nms_threshold,
            keep_pre_nms=keep_pre_nms, keep_post_nms=keep_post_nms,
            dtype=dtype, device=device)

        if targets is not None:
            # merge batches
            cls_logits = cls_logits.reshape(-1)
            reg_deltas = reg_deltas.reshape(-1,4)
                        
            # match/build targets
            matches,target_objectness,target_labels,target_offsets = build_targets(
                self.detection_layer.anchors.repeat(bs,1,1),
                targets, fg_iou_threshold, bg_iou_threshold,
                add_best_matches=True)

            # sample fg and bg with given ratio
            positives,negatives = sample_fg_bg(matches,batch_size_per_image,batch_positive_ratio)
            samples = torch.cat([positives,negatives])

            # compute loss
            cls_loss,reg_loss = self.compute_loss(
                cls_logits[samples], target_objectness[samples],
                reg_deltas[positives], target_offsets[positives])

            losses = {'cls_loss': cls_loss,'reg_loss': reg_loss}

            return batched_dets,losses

        return batched_dets

    def compute_loss(self,
        objectness:torch.Tensor, gt_objectness:torch.Tensor,
        deltas:torch.Tensor, gt_deltas:torch.Tensor):

        num_samples = objectness.size(0)

        cls_loss = F.binary_cross_entropy_with_logits(objectness, gt_objectness)
        reg_loss = F.smooth_l1_loss(deltas, gt_deltas, reduction='sum') / num_samples
        return cls_loss,reg_loss


class RPN(nn.Module):
    def __init__(self, backbone,
        rpn_kernel_size:int=3,
        anchor_sizes:List[int]=[128,256,512], anchor_aspect_ratios:List[float]=[0.5,1,2],
        nms_threshold:float=0.7,
        train_keep_pre_nms:int=2000, train_keep_post_nms:int=2000,
        test_keep_pre_nms:int=1000, test_keep_post_nms:int=300,
        batch_size_per_image:int=256, batch_positive_ratio:float=0.5,
        fg_iou_threshold:float=0.7, bg_iou_threshold:float=0.3,
        anchor_generator=None):

        super(RPN,self).__init__()
        self.backbone = backbone
        assert hasattr(backbone,'output_channels'),"backbone must have output channels variable as integer"

        self.head = RPNHead(backbone.output_channels,
            rpn_kernel_size=rpn_kernel_size,
            anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_aspect_ratios,
            nms_threshold=nms_threshold,
            train_keep_pre_nms=train_keep_pre_nms, train_keep_post_nms=train_keep_post_nms,
            test_keep_pre_nms=test_keep_pre_nms, test_keep_post_nms=test_keep_post_nms,
            batch_size_per_image=batch_size_per_image, batch_positive_ratio=batch_positive_ratio,
            fg_iou_threshold=fg_iou_threshold, bg_iou_threshold=bg_iou_threshold,
            anchor_generator=anchor_generator)

    def forward(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]=None):

        img_dims = batch.shape[-2:]

        fmaps = self.backbone(batch)

        batched_rois,losses = self.head(fmaps, img_dims, targets=targets)

        return batched_rois,losses

    def training_step(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
        batched_rois,losses = self.forward(batch, targets=targets)

        joint_loss = losses['cls_loss'] + losses['reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()

        losses['loss'] = joint_loss

        return losses

    def validation_step(self, batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
        with torch.no_grad():
            batched_rois,losses = self.forward(batch, targets=targets)

        roi_targets = [target['boxes'].cpu() for target in targets] # K,4
        losses['loss'] = losses['cls_loss'] + losses['reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()

        detections = {
            'predictions': [rois.cpu() for rois in batched_rois],
            'ground_truths': roi_targets
        }
        
        return detections,losses