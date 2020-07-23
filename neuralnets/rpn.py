import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,List,Tuple
from torchvision.ops import nms
from cv2 import cv2
from utils import (
    generate_anchors,
    generate_default_boxes,
    apply_box_regressions,
    clip_boxes,
    get_ignore_mask,
    build_targets,
    resample_pos_neg_distribution
)

class DetectionLayer(nn.Module):
    def __init__(self, effective_stride:int=16,
            anchor_scales:List=[0.5,1,2], anchor_ratios:List=[0.5,1,2]):
        super(DetectionLayer,self).__init__()

        self.effective_stride = effective_stride

        self.anchors = generate_anchors(self.effective_stride,
            ratios=anchor_ratios, scales=anchor_scales)

        self._fw = 62
        self._fh = 37
        self.default_boxes = generate_default_boxes(
            self.anchors, (self._fh, self._fw), self.effective_stride)

    def forward(self, preds:torch.Tensor, regs:torch.Tensor, img_dims:Tuple, iou_threshold:float=0.7,
            conf_threshold:float=0.5, keep_pre_nms:int=6000, keep_post_nms:int=300) -> List[torch.Tensor]:
        """Computes region of interests

        Args:
            preds (torch.Tensor): bs x fmap_h x fmap_w x num_anchors
            regs (torch.Tensor): bs x fmap_h x fmap_w x num_anchors x 4
            img_dims (Tuple): height,width of the image
            iou_threshold (float, optional): [description]. Defaults to 0.7.
            conf_threshold (float, optional): [description]. Defaults to 0.5.
            keep_pre_nms (int, optional): [description]. Defaults to 6000.
            keep_post_nms (int, optional): [description]. Defaults to 300.

        Returns:
            List[torch.Tensor]:[xmin,ymin,xmax,ymax,score]
        """

        bs,fh,fw,_ = preds.shape

        if preds.device != self.anchors.device:
            self.anchors = self.anchors.to(preds.device)

        if fw != self._fw or fh != self._fh:
            # re-generate default boxes if feature map size is changed
            self._fw = fw
            self._fh = fh
            self.default_boxes = generate_default_boxes( # fh x fw x nA x 4 as xmin ymin xmax ymax
                self.anchors, (self._fh, self._fw),
                self.effective_stride, device=preds.device)

        scores = torch.sigmoid(preds).reshape(bs,-1)

        boxes = apply_box_regressions(regs.reshape(bs,-1,4), self.default_boxes.reshape(-1,4))
        boxes = clip_boxes(boxes, img_dims)

        rois = []
        for i in range(bs):
            sort = scores[i].argsort(dim=-1, descending=True)
            proposals = boxes[i][sort][:keep_pre_nms,:]
            sc = scores[i][sort]
            pick = nms(proposals, sc, iou_threshold)

            proposals = proposals[pick]
            sc = sc[pick]
            proposals = torch.cat([proposals[:keep_post_nms], sc[:keep_post_nms].unsqueeze(-1)],dim=1)
            pick = proposals[:,4] > conf_threshold
            rois.append(proposals[pick,:4])

        return rois

class RPN(nn.Module):
    """Input features represents vgg16 backbone and n=3 is set because of the `faster rcnn` parper

    Number of anchors selected from paper, where num_anchors: num_scales * num_ratios
    """
    def __init__(self, features:int=512, n:int=3, effective_stride:int=16,
            anchor_scales:List=[0.5,1,2], anchor_ratios:List=[0.5,1,2],
            train_iou_threshold:float=0.7, train_conf_threshold:float=0.0,
            train_keep_pre_nms:int=6000, train_keep_post_nms:int=2000,
            test_iou_threshold:float=0.7, test_conf_threshold:float=0.1,
            test_keep_pre_nms:int=6000, test_keep_post_nms:int=300,
            num_of_samples:int=256, positive_ratio:float=.5):

        super(RPN,self).__init__()

        assert n % 2 == 1,"kernel size must be odd"

        num_anchors = len(anchor_ratios) * len(anchor_scales)

        # padding calculation for same input output dimensions
        padding = int((n-1) / 2)

        self.base_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features,
                kernel_size=n, stride=1, padding=padding),
            nn.ReLU(inplace=True))

        self.cls_conv_layer = nn.Conv2d(
            in_channels=features, out_channels=num_anchors,
            kernel_size=1, stride=1, padding=0)

        self.reg_conv_layer = nn.Conv2d(
            in_channels=features, out_channels=num_anchors*4,
            kernel_size=1, stride=1, padding=0)

        self.detection_layer = DetectionLayer(effective_stride=effective_stride,
            anchor_ratios=anchor_ratios, anchor_scales=anchor_scales)

        self._params = {
            'train':{
                'iou_threshold':train_iou_threshold,
                'conf_threshold':train_conf_threshold,
                'keep_pre_nms':train_keep_pre_nms,
                'keep_post_nms':train_keep_post_nms
            },
            'test':{
                'iou_threshold':test_iou_threshold,
                'conf_threshold':test_conf_threshold,
                'keep_pre_nms':test_keep_pre_nms,
                'keep_post_nms':test_keep_post_nms
            },
            'positive_ratio': positive_ratio,
            'num_of_samples': num_of_samples
        }

    def get_params(self):
        return self._params['train'] if self.training else self._params['test']

    def forward(self, fmap:torch.Tensor, img_dims:Tuple, targets:List[Dict[str,torch.Tensor]]=None):
        output = self.base_conv_layer(fmap)
        preds = self.cls_conv_layer(output)
        regs = self.reg_conv_layer(output)

        bs,_,fh,fw = preds.shape

        # bs x num_anchors x fmap_h x fmap_w => bs x fmap_h x fmap_w x num_anchors
        preds = preds.permute(0,2,3,1)

        # bs x num_anchors*4 x fmap_h x fmap_w => bs x fmap_h x fmap_w x (num_anchors*4)
        regs = regs.permute(0,2,3,1).reshape(bs,fh,fw,-1,4)

        params = self.get_params()

        rois = self.detection_layer(preds,regs,img_dims, **params)

        if targets is not None:
            losses = self.compute_loss(preds, regs, img_dims, targets)
            # alter rois here
            return rois,losses

        return rois

    def compute_loss(self, preds:torch.Tensor, regs:torch.Tensor,
            img_dims:Tuple, targets:List[Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:
        """Computes losses for cls and reg

        Args:
            preds (torch.Tensor): bs x fmap_h x fmap_w x num_anchors
            regs (torch.Tensor): bs x fmap_h x fmap_w x num_anchors x 4
            img_dims (Tuple): height,width
            targets (List[Dict[str,torch.Tensor]]): [
                {
                    'boxes':torch.Tensor(M,4), # as xmin,ymin,xmax,ymax
                    'labels':torch.Tensor(M,)  # as torch.int64 (Long)
                },
                {
                    'boxes':torch.Tensor(M,4), # as xmin,ymin,xmax,ymax
                    'labels':torch.Tensor(M,)  # as torch.int64 (Long)
                },
                ...
            ]

        Returns:
            Dict[str,torch.Tensor]: {'cls_loss', cls_loss, 'reg_loss': reg_loss}
        """

        ignore_mask = get_ignore_mask(self.detection_layer.default_boxes, img_dims)

        # * t_preds not used since classes only contains {BG|FG} for RPN
        (pos_mask,neg_mask),(t_objness,t_preds,t_regs) = build_targets(
            preds,regs, self.detection_layer.default_boxes, ignore_mask, targets)

        # resample positive and negatives to have ratio of 1:1 (pad with negatives if needed)
        pos_mask,neg_mask = resample_pos_neg_distribution(
            pos_mask, neg_mask, total_count=self._params['num_of_samples'],
            positive_ratio=self._params['positive_ratio'])

        # calculate binary cross entropy with logits
        cls_loss = F.binary_cross_entropy_with_logits(
            preds[pos_mask | neg_mask], t_objness[pos_mask | neg_mask], reduction='sum')

        # calculate smooth l1 loss for bbox regression
        reg_loss = F.smooth_l1_loss(regs[pos_mask][:,0], t_regs[pos_mask][:,0], reduction='sum') +\
            F.smooth_l1_loss(regs[pos_mask][:,1], t_regs[pos_mask][:,1], reduction='sum') +\
                F.smooth_l1_loss(regs[pos_mask][:,2], t_regs[pos_mask][:,2], reduction='sum') +\
                    F.smooth_l1_loss(regs[pos_mask][:,3], t_regs[pos_mask][:,3], reduction='sum')

        return {'cls_loss':cls_loss,'reg_loss':reg_loss}