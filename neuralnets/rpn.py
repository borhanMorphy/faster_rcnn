import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,List,Tuple
from torchvision.ops import boxes as box_ops
from cv2 import cv2

from utils.boxv2 import offsets2boxes, AnchorGenerator, boxes2offsets

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

    def forward(self, fmaps:List[torch.Tensor]) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:
        """
        Params:
            fmaps List[torch.Tensor]: list of feature maps with shape 1 x c' x h' x w'

        Returns:
            Tuple[List[torch.Tensor],List[torch.Tensor]]
                cls_logits: [(1 x (h'*w'*nA) x 1), ...]
                reg_deltas: [(1 x (h'*w'*nA) x 4), ...] as dx,dy,dw,dh
        """
        cls_logits:List[torch.Tensor] = []
        reg_deltas:List[torch.Tensor] = []
        for i in range(len(fmaps)):
            assert fmaps[i].size(0) == 1,"only 1 batch size allowed for now"
            output = F.relu(self.base_conv_layer(fmaps[i]))
            logits = self.cls_conv_layer(output)
            logits = self.flatten_preds(logits, 1)

            deltas = self.reg_conv_layer(output)
            deltas = self.flatten_preds(deltas, 4)

            cls_logits.append( logits )
            reg_deltas.append( deltas )

        return cls_logits,reg_deltas

class DetectionLayer(nn.Module):
    def __init__(self, anchor_generator):
        super(DetectionLayer,self).__init__()
        self.anchor_generator = anchor_generator

    @torch.no_grad()
    def forward(self, cls_logits:List[torch.Tensor], reg_deltas:List[torch.Tensor],
            fmap_dims:List[Tuple[int,int]], img_dims:List[Tuple[int,int]],
            nms_threshold:float=.7, keep_pre_nms:int=1000, keep_post_nms:int=300,
            dtype=torch.float32, device='cpu'):
        """
        Params:
            cls_logits List[torch.Tensor]: [torch.Tensor(1 x (h'*w'*nA) x 1), ...] 
            reg_deltas List[torch.Tensor]: [torch.Tensor(1 x (h'*w'*nA) x 4), ...]
            fmap_dims:List[Tuple[int,int]] [(h',w'), ...]
            img_dims:List[Tuple[int,int]] [(h,w), ...]

        Returns:
            batched_dets: List[torch.Tensor(N,5)] as xmin,ymin,xmax,ymax,score
        """
        # generate anchors for each input
        self.batch_anchors = self.anchor_generator(fmap_dims, img_dims, dtype=dtype, device=device)

        batch_iter = zip(cls_logits,reg_deltas,self.batch_anchors,img_dims)
        batched_dets:List[torch.Tensor] = []

        for cls_logit,reg_delta,anchors,img_dim in batch_iter:
            # N = h'*w'*nA
            # cls_logit: torch.Tensor(N,)
            # reg_delta: torch.Tensor(N,4)
            # anchors: torch.Tensor(N,4)
            # img_dim: (h,w)
            
            # torch.Tensor(N,) => torch.Tensor(N,)
            scores = torch.sigmoid(cls_logit.detach()).reshape(-1) # ! assumed bs=1
            offsets = reg_delta.detach().reshape(-1,4) # ! assumed bs=1

            # convert offsets to boxes
            # N,4 | N,4 => N,4 as xmin,ymin,xmax,ymax
            boxes = offsets2boxes(offsets, anchors)

            # select top n
            _,selected_ids = scores.topk(keep_pre_nms)
            scores,boxes = scores[selected_ids], boxes[selected_ids]

            # clip boxes
            boxes = box_ops.clip_boxes_to_image(boxes, img_dim)

            # remove small
            keep = box_ops.remove_small_boxes(boxes, 1e-3) # TODO try 1
            scores,boxes = scores[keep], boxes[keep]

            # nms
            keep = box_ops.nms(boxes, scores, nms_threshold)
            scores,boxes = scores[keep], boxes[keep]

            # post_n
            keep_post_nms = min(keep_post_nms, boxes.size(0))
            scores,boxes = scores[:keep_post_nms], boxes[:keep_post_nms]

            # ! warning 0 size might be cause error
            batched_dets.append( torch.cat([boxes,scores.unsqueeze(-1)], dim=-1) )

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

    @staticmethod
    def build_targets(batched_anchors:List[torch.Tensor],
            targets:List[Dict[str,torch.Tensor]],
            pos_iou_tresh:float, neg_iou_tresh:float,
            add_best_matches:bool=True) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Params:

        Returns:
            matches torch.Tensor: Ntotal,
            target_objectness torch.Tensor : Ntotal, 
            target_offsets torch.Tensor : Ntotal, 4
        """
        batch_matches:List[torch.Tensor] = []
        batch_target_objectness:List[torch.Tensor] = []
        batch_target_offsets:List[torch.Tensor] = []

        for anchors,targets_per_image in zip(batched_anchors,targets):
            device = anchors.device
            dtype = anchors.dtype

            gt_boxes = targets_per_image['boxes'].to(device, dtype)
            # ignored due to only objectness score is required
            gt_labels = targets_per_image['labels'].to(device, torch.int64)

            N = anchors.size(0)
            M = gt_boxes.size(0)
            # -1: negative match | 0: ignore | 1: positive match
            matches = torch.zeros(N, dtype=dtype, device=device)
            target_objectness = torch.zeros(N, dtype=dtype, device=device)
            target_offsets = torch.zeros(N,4 , dtype=dtype, device=device)

            if M == 0:
                batch_matches.append(matches)
                batch_target_objectness.append(target_objectness)
                batch_target_offsets.append(target_offsets)
                continue

            ious = box_ops.box_iou(anchors,gt_boxes) # N,M
            # best_values: N,
            # best_matches: N,
            best_values,best_matches = ious.max(dim=1)
            # best_anchor_match_ids: M,
            _, best_anchor_match_ids = ious.max(dim=0)

            matches[best_values >= pos_iou_tresh] = 1
            matches[best_values <= neg_iou_tresh] = -1

            if add_best_matches: matches[best_anchor_match_ids] = 1

            for i in range(M):
                box = gt_boxes[i]
                mask = best_matches == i
                target_offsets[mask] = box

            # convert boxes to offsets
            target_offsets = boxes2offsets(target_offsets, anchors)

            # set fg label for objectness
            target_objectness[ matches == 1 ] = 1 

            batch_matches.append(matches)
            batch_target_objectness.append(target_objectness)
            batch_target_offsets.append(target_offsets)

        batch_matches = torch.cat(batch_matches, dim=0)
        batch_target_objectness = torch.cat(batch_target_objectness, dim=0)
        batch_target_offsets = torch.cat(batch_target_offsets, dim=0)

        return batch_matches, batch_target_objectness, batch_target_offsets

    @staticmethod
    def sample_fg_bg(matches, total_samples:int, positive_ratio:float):
        positives, = torch.where(matches == 1)
        negatives, = torch.where(matches == -1)

        num_pos = positives.size(0)
        num_neg = negatives.size(0)

        num_pos = min(int(total_samples * positive_ratio), num_pos)
        num_neg = min(total_samples-num_pos, num_neg)

        positives = torch.randperm(positives.size(0), device=positives.device)[:num_pos]
        negatives = torch.randperm(negatives.size(0), device=negatives.device)[:num_neg]

        return positives,negatives

    def get_params(self):
        if self.training:
            return self._params['train'],self._params['others']
        return self._params['test'],self._params['others']

    def forward(self, fmaps:List[torch.Tensor], img_dims:List[Tuple[int,int]],
            targets:List[Dict[str,torch.Tensor]]=None):
        (keep_pre_nms, keep_post_nms),\
            (batch_size_per_image, batch_positive_ratio,\
                fg_iou_threshold, bg_iou_threshold,\
                    nms_threshold) = self.get_params()

        dtype = fmaps[0].dtype
        device = fmaps[0].device

        fmap_dims = [fmap.shape[-2:] for fmap in fmaps]

        # cls_logits: [(1 x (h'*w'*nA) x 1), ...]
        # reg_deltas: [(1 x (h'*w'*nA) x 4), ...] as dx,dy,dw,dh
        cls_logits,reg_deltas = self.prediction_layer(fmaps)

        batched_dets = self.detection_layer(cls_logits, reg_deltas, fmap_dims,
            img_dims, nms_threshold=nms_threshold,
            keep_pre_nms=keep_pre_nms, keep_post_nms=keep_post_nms,
            dtype=dtype, device=device)

        if targets is not None:
            cls_logits = torch.cat(cls_logits, dim=1).reshape(-1)
            reg_deltas = torch.cat(reg_deltas, dim=1).reshape(-1,4)
                        
            # match/build targets
            matches,target_objectness,target_offsets = self.build_targets(self.detection_layer.batch_anchors,
                targets, fg_iou_threshold, bg_iou_threshold)

            # sample fg and bg with given ratio
            positives,negatives = self.sample_fg_bg(matches,batch_size_per_image,batch_positive_ratio)
            samples = torch.cat([positives,negatives])

            # compute loss
            cls_loss,reg_loss = self.compute_loss(
                cls_logits[samples], target_objectness[samples],
                reg_deltas[positives], target_offsets[positives])

            losses = {
                'rpn_cls_loss': cls_loss,
                'rpn_reg_loss': reg_loss}

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

    def forward(self, images:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]=None):

        img_dims = [img.shape[-2:] for img in images]

        fmaps = [self.backbone(img) for img in images]

        batched_rois,losses = self.head(fmaps, img_dims, targets=targets)

        return batched_rois,losses

    def training_step(self, images:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]):
        batched_rois,losses = self.forward(images, targets=targets)

        joint_loss = losses['rpn_cls_loss'] + losses['rpn_reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()

        losses['loss'] = joint_loss

        return losses

    @torch.no_grad()
    def validation_step(self, images:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]):
        batched_rois,losses = self.forward(images, targets=targets)

        roi_targets = [target['boxes'].cpu() for target in targets] # K,4
        losses['loss'] = losses['rpn_cls_loss'] + losses['rpn_reg_loss']

        for k in losses:
            losses[k] = losses[k].detach().item()

        detections = {
            'predictions': [rois.cpu() for rois in batched_rois],
            'ground_truths': roi_targets
        }
        
        return detections,losses