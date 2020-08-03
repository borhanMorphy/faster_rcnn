import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool,MultiScaleRoIAlign
from typing import List,Dict,Tuple
from torchvision.ops import boxes as box_ops
from collections import OrderedDict
from utils import (
    build_targets,
    sample_fg_bg,
    offsets2boxes
)

class FastRCNNHead(nn.Module):
    def __init__(self, features:int, num_classes:int,
            roi_output_size:int=7, hidden_channels:int=1024,
            batch_size_per_image:int=512, batch_positive_ratio:float=.25,
            fg_iou_threshold:float=0.5, bg_iou_threshold:float=0.5,
            conf_threshold:float=0.05, nms_threshold:float=0.5, keep_top_n:int=100):
        super(FastRCNNHead,self).__init__()

        self.num_classes = num_classes
        self.roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=roi_output_size,
                sampling_ratio=2)
        self.hidden_unit = nn.Sequential(
            nn.Linear(
                in_features=roi_output_size*roi_output_size*features,
                out_features=hidden_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hidden_channels,
                out_features=hidden_channels, bias=True),
            nn.ReLU(inplace=True))

        self.cls_unit = nn.Linear(hidden_channels, num_classes)
        self.reg_unit = nn.Linear(hidden_channels, num_classes*4)

        self._params = {
            'fg_iou_threshold': fg_iou_threshold,
            'bg_iou_threshold': bg_iou_threshold,
            'num_of_samples': batch_size_per_image,
            'positive_ratio': batch_positive_ratio,
            'conf_threshold': conf_threshold,
            'nms_threshold': nms_threshold,
            'keep_top_n': keep_top_n}

    def forward(self, fmaps:Dict[str,torch.Tensor], rois:List[torch.Tensor], img_dims:List[Tuple[int,int]],
            targets:List[Dict[str,torch.Tensor]]=None):

        # assign rois to gt and generate cls(Ntotal,) and reg(Ntotal,4) targets
        if targets is not None:
            # match/build targets
            matches,target_objectness,target_labels,target_offsets = build_targets(
                rois, targets,
                self._params['fg_iou_threshold'],self._params['bg_iou_threshold'],
                add_best_matches=False)

            # sample fg and bg with given ratio
            positives,negatives = sample_fg_bg(matches,
                self._params['num_of_samples'], self._params['positive_ratio'])

            matches[:] = 0 
            matches[positives] = 1
            matches[negatives] = -1
            sample_mask = torch.logical_or(matches == 1,matches == -1)
            positive_mask = matches == 1

            target_objectness = target_objectness[sample_mask]
            target_labels = target_labels[sample_mask]
            target_offsets = target_offsets[positive_mask]
            current = 0
            for roi_index,boxes in enumerate(rois):
                N = boxes.size(0)
                batch_sample_mask = sample_mask[current: current+N]
                current += N
                rois[roi_index] = boxes[batch_sample_mask]
            

        # extract all rois from feature maps (Ntotal,(C*output_size[0]*output_size[1]))
        # outputs: (Ntotal,output_features*output_size**2)
    
        outputs = self.roi_pool(fmaps, rois, img_dims).flatten(start_dim=1)

        # feed to the hidden units and get cls_logits and reg_deltas

        outputs = self.hidden_unit(outputs) # Ntotal,hiddin_channels
        cls_logits = self.cls_unit(outputs) # Ntotal,num_classes
        reg_deltas = self.reg_unit(outputs)  # Ntotal,num_classes*4
        reg_deltas = reg_deltas.reshape(-1,self.num_classes,4)

        return (cls_logits,reg_deltas),(target_objectness,target_labels,target_offsets)


class FastRCNNMultiHead(FastRCNNHead):
    def __init__(self, features:int, num_classes:int, **kwargs):
        super(FastRCNNMultiHead,self).__init__(features,num_classes,**kwargs)

    def forward(self, fmaps:Dict[str,torch.Tensor], rois:List[torch.Tensor], img_dims:List[Tuple[int,int]],
            targets:List[Dict[str,torch.Tensor]]=None):

        (cls_logits,reg_deltas),(target_objectness,target_labels,target_offsets) = super().forward(
            fmaps,rois,img_dims,targets=targets)

        batched_dets = self.post_process(cls_logits,reg_deltas,rois)

        if targets is not None:
            pos_mask = target_labels != 0
            reg_deltas = reg_deltas[pos_mask, target_labels[pos_mask]]

            # compute loss
            cls_loss,reg_loss = self.compute_loss(
                cls_logits, target_labels,
                reg_deltas, target_offsets)

            losses = {'cls_loss': cls_loss,'reg_loss': reg_loss}

            return batched_dets,losses

        return batched_dets

    def post_process(self, cls_logits:torch.Tensor,
            reg_deltas:torch.Tensor, batched_rois:List[torch.Tensor]):
        nms_threshold = self._params['nms_threshold']
        conf_threshold = self._params['conf_threshold']
        keep_top_n = self._params['keep_top_n']

        batched_dets:List[torch.Tensor] = []
        current = 0
        for rois in batched_rois:
            N = rois.size(0)
            if N == 0:
                print("warning! found empty rois")
                batched_dets.append( torch.empty(0,6, dtype=reg_deltas.dtype, device=reg_deltas.device) )
                continue

            logits = cls_logits[current:current+N]
            offsets = reg_deltas[current:current+N]
            current += N

            # logits: torch.Tensor(N,)
            # deltas: torch.Tensor(N,4)
            # rois: torch.Tensor(N,4)

            scores,preds = F.softmax(logits, dim=1).max(dim=1)

            fg_preds_mask = preds != 0

            # N,nC,4 => N,4
            offsets = offsets.gather(1, preds.view(-1,1).repeat(1,4).unsqueeze(1)).squeeze(1)

            # convert offsets to boxes
            # N,4 | N,4 => N,4 as xmin,ymin,xmax,ymax

            boxes = offsets2boxes(offsets.unsqueeze(0), rois).squeeze(0)

            # extract bg predictions
            boxes = boxes[fg_preds_mask]
            preds = preds[fg_preds_mask]
            scores = scores[fg_preds_mask]

            # apply conf threshold
            keep = scores >= conf_threshold
            scores,preds,boxes = scores[keep],preds[keep],boxes[keep]

            # remove small
            keep = box_ops.remove_small_boxes(boxes, 1e-3) # TODO try 1
            scores,preds,boxes = scores[keep],preds[keep],boxes[keep]

            # batched nms
            keep = box_ops.batched_nms(boxes.cpu(), scores.cpu(), preds.cpu(), nms_threshold)
            scores,preds,boxes = scores[keep],preds[keep],boxes[keep]

            # select top n
            keep_n = min(keep_top_n,scores.size(0))
            _,selected_ids = scores.topk(keep_n)
            scores,preds,boxes = scores[selected_ids],preds[selected_ids],boxes[selected_ids]
            scores.unsqueeze_(1)
            preds = preds.unsqueeze(1).to(boxes.dtype)
            dets = torch.cat([boxes,scores,preds], dim=-1)

            batched_dets.append(dets)

        return batched_dets

    def compute_loss(self,
            cls_logits:torch.Tensor, gt_labels:torch.Tensor,
            deltas:torch.Tensor, gt_deltas:torch.Tensor):

        num_samples = cls_logits.size(0)

        cls_loss = F.cross_entropy(cls_logits, gt_labels)
        reg_loss = F.smooth_l1_loss(deltas, gt_deltas, reduction='sum') / num_samples
        return cls_loss,reg_loss


class FastRCNNSingleHead(FastRCNNHead):
    def __init__(self, features:int, **kwargs):
        super(FastRCNNSingleHead,self).__init__(features,1,**kwargs)

    def forward(self, fmaps:Dict[str,torch.Tensor], rois:List[torch.Tensor], img_dims:List[Tuple[int,int]],
            targets:List[Dict[str,torch.Tensor]]=None):

        (cls_logits,reg_deltas),(target_objectness,target_labels,target_offsets) = super().forward(
            fmaps,rois,img_dims,targets=targets)

        cls_logits = cls_logits.reshape(-1)
        reg_deltas = reg_deltas.reshape(-1,4)

        batched_dets = self.post_process(cls_logits,reg_deltas,rois)

        if targets is not None:
            pos_mask = target_objectness != 0
            reg_deltas = reg_deltas[pos_mask]

            # compute loss
            cls_loss,reg_loss = self.compute_loss(
                cls_logits, target_objectness,
                reg_deltas, target_offsets)

            losses = {'cls_loss': cls_loss,'reg_loss': reg_loss}

            return batched_dets,losses

        return batched_dets

    def post_process(self, cls_logits:torch.Tensor,
            reg_deltas:torch.Tensor, batched_rois:List[torch.Tensor]):
        nms_threshold = self._params['nms_threshold']
        conf_threshold = self._params['conf_threshold']
        keep_top_n = self._params['keep_top_n']

        batched_dets:List[torch.Tensor] = []
        current = 0
        for rois in batched_rois:
            N = rois.size(0)
            if N == 0:
                print("warning! found empty rois")
                batched_dets.append( torch.empty(0,6, dtype=reg_deltas.dtype, device=reg_deltas.device) )
                continue

            logits = cls_logits[current:current+N]
            offsets = reg_deltas[current:current+N]
            current += N

            # logits: torch.Tensor(N,)
            # deltas: torch.Tensor(N,4)
            # rois: torch.Tensor(N,4)

            scores = torch.sigmoid(logits)
            preds = torch.zeros(scores.shape, dtype=torch.int64, device=scores.device)
            preds[scores >= 0.5] = 1

            fg_preds_mask = preds != 0

            # convert offsets to boxes
            # N,4 | N,4 => N,4 as xmin,ymin,xmax,ymax

            boxes = offsets2boxes(offsets.unsqueeze(0), rois).squeeze(0)

            # extract bg predictions
            boxes = boxes[fg_preds_mask]
            preds = preds[fg_preds_mask]
            scores = scores[fg_preds_mask]

            # apply conf threshold
            keep = scores >= conf_threshold
            scores,preds,boxes = scores[keep],preds[keep],boxes[keep]

            # remove small
            keep = box_ops.remove_small_boxes(boxes, 1e-3) # TODO try 1
            scores,preds,boxes = scores[keep],preds[keep],boxes[keep]

            # batched nms
            keep = box_ops.batched_nms(boxes.cpu(), scores.cpu(), preds.cpu(), nms_threshold)
            scores,preds,boxes = scores[keep],preds[keep],boxes[keep]

            # select top n
            keep_n = min(keep_top_n,scores.size(0))
            _,selected_ids = scores.topk(keep_n)
            scores,preds,boxes = scores[selected_ids],preds[selected_ids],boxes[selected_ids]
            scores.unsqueeze_(1)
            preds = preds.unsqueeze(1).to(boxes.dtype)
            dets = torch.cat([boxes,scores,preds], dim=-1)

            batched_dets.append(dets)

        return batched_dets

    def compute_loss(self,
            cls_logits:torch.Tensor, gt_labels:torch.Tensor,
            deltas:torch.Tensor, gt_deltas:torch.Tensor):

        num_samples = cls_logits.size(0)

        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, gt_labels)
        reg_loss = F.smooth_l1_loss(deltas, gt_deltas, reduction='sum') / num_samples
        return cls_loss,reg_loss

class FastRCNN(nn.Module):
    def __init__(self, backbone, num_classes:int,
            output_size:int, hidden_channels:int=1024,
            batch_size_per_image:int=512, batch_positive_ratio:float=.25,
            fg_iou_threshold:float=0.5, bg_iou_threshold:float=0.5,
            conf_threshold:float=0.05, nms_threshold:float=0.5, keep_top_n:int=100):

        super(FastRCNN,self).__init__()
        assert hasattr(backbone,'output_channels'),"backbone must have output channels variable as integer"
        assert num_classes > 1,"must be greater than 1 since includes background class"

        self.backbone = backbone
        features = backbone.output_channels

        if num_classes > 2:
            self.head = FastRCNNMultiHead(features, num_classes,
                roi_output_size=output_size, hidden_channels=hidden_channels,
                batch_size_per_image=batch_size_per_image, batch_positive_ratio=batch_positive_ratio,
                fg_iou_threshold=fg_iou_threshold, bg_iou_threshold=bg_iou_threshold,
                conf_threshold=conf_threshold, nms_threshold=nms_threshold, keep_top_n=keep_top_n)
        else:
            self.head = FastRCNNSingleHead(features,
                roi_output_size=output_size, hidden_channels=hidden_channels,
                batch_size_per_image=batch_size_per_image, batch_positive_ratio=batch_positive_ratio,
                fg_iou_threshold=fg_iou_threshold, bg_iou_threshold=bg_iou_threshold,
                conf_threshold=conf_threshold, nms_threshold=nms_threshold, keep_top_n=keep_top_n)


    def forward(self, images:List[torch.Tensor], rois:List[torch.Tensor],
            targets:List[Dict[str,torch.Tensor]]=None):

        fmaps = [self.backbone(img) for img in images]

        dets = self.head(fmaps,rois,targets=targets)

        return dets