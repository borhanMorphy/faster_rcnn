import torch.nn as nn
from typing import Dict,List,Tuple
import torch
from torchvision.ops import nms
import torch.nn.functional as F
from cv2 import cv2

from utils.box import (
    generate_anchors,
    generate_default_boxes,
    apply_box_regressions,
    clip_boxes,
    get_ignore_mask
)

from utils.train import (
    build_targets,
    build_targets_v2,
    resample_pos_neg_distribution
)
from utils.data import load_data


class DetectionLayer(nn.Module):
    def __init__(self, effective_stride:int=16,
            anchor_scales:List=[0.5,1,2], anchor_ratios:List=[0.5,1,2],
            conf_threshold:float=0.5, iou_threshold:float=0.7, keep_n:int=300):
        super(DetectionLayer,self).__init__()

        self.effective_stride = effective_stride
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.keep_n = keep_n

        self.anchors = generate_anchors(self.effective_stride,
            ratios=anchor_ratios, scales=anchor_scales)

        self._fw = 62
        self._fh = 37
        self.default_boxes = generate_default_boxes(
            self.anchors, (self._fh, self._fw), self.effective_stride)

    def forward(self, preds:torch.Tensor, regs:torch.Tensor) -> List[torch.Tensor]:

        bs,_,fh,fw = preds.shape
        # bs x num_anchors x fmap_h x fmap_w => bs x fmap_h x fmap_w x num_anchors
        preds = preds.permute(0,2,3,1)

        # bs x num_anchors*4 x fmap_h x fmap_w => bs x fmap_h x fmap_w x (num_anchors*4)
        regs = regs.permute(0,2,3,1).reshape(bs,fh,fw,-1,4)

        if preds.device != self.anchors.device:
            self.anchors = self.anchors.to(preds.device)

        if fw != self._fw or fh != self._fh:
            # re-generate default boxes if feature map size is changed
            self._fw = fw
            self._fh = fh
            self.default_boxes = generate_default_boxes( # fh x fw x nA x 4 as xmin ymin xmax ymax
                self.anchors, (self._fh, self._fw),
                self.effective_stride, device=preds.device)
        return preds,regs

    def _inference_postprocess(self, preds:torch.Tensor, regs:torch.Tensor, img_dims:torch.Tensor):
        """
        Arguments:
            preds: bs x (fmap_h * fmap_w * nA)
            regs:  bs x (fmap_h * fmap_w * nA)  x 4
            img_dims: torch.tensor(height,width)
        """
        bs = preds.size(0)

        # bs x N x 2 => bs x N
        # TODO convert to sigmoid since its bce loss
        scores = F.softmax(preds, dim=-1)[:,:,1]


        # bs x N x 4
        # ! here is something wrong
        boxes = apply_box_regressions(self.default_boxes, regs)

        boxes = clip_boxes(boxes, img_dims)

        det_results = []
        for i in range(bs):
            sort = scores[i].argsort(dim=-1, descending=True)
            proposals = boxes[i][sort][:6000,:]
            sc = scores[i][sort]
            pick = nms(proposals, sc, self.iou_threshold)

            proposals = proposals[pick]
            sc = sc[pick]
            proposals = torch.cat([proposals[:self.keep_n], sc[:self.keep_n].unsqueeze(-1)],dim=1)
            pick = proposals[:,4] > self.conf_threshold
            det_results.append(proposals[pick,:])

        return det_results

class RPN(nn.Module):
    """Input features represents vgg16 backbone and n=3 is set because of the `faster rcnn` parper

    Number of anchors selected from paper, where num_anchors: num_scales * num_ratios
    """
    def __init__(self, backbone, features:int=512, n:int=3, effective_stride:int=16,
            anchor_scales:List=[0.5,1,2], anchor_ratios:List=[0.5,1,2],
            conf_threshold:float=0.5, iou_threshold:float=0.7, keep_n:int=300):
        super(RPN,self).__init__()

        self.backbone = backbone
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
            anchor_ratios=anchor_ratios, anchor_scales=anchor_scales,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold, keep_n=keep_n)

        self.debug = True

    def forward(self, batch:torch.Tensor):
        fmap = self.backbone(batch)
        output = self.base_conv_layer(fmap)

        preds = self.cls_conv_layer(output)
        regs = self.reg_conv_layer(output)

        return self.detection_layer(preds,regs)

    def training_step(self, batch, targets, imgs=None):
        ih,iw = batch.shape[-2:]
        preds,regs = self.forward(batch)
        ignore_mask = get_ignore_mask(self.detection_layer.default_boxes, (ih,iw))

        Ncls = 256
        Nreg = 2400
        w_lamda = 10
        total_samples = 256
        pos_ratio = .5

        # * t_preds not used since classes only contains {BG|FG} for RPN
        (pos_mask,neg_mask),(t_objness,t_preds,t_regs) = build_targets_v2(
            preds,regs, self.detection_layer.default_boxes, ignore_mask, targets, imgs=imgs)

        # resample positive and negatives to have ratio of 1:1 (pad with negatives if needed)
        pos_mask,neg_mask = resample_pos_neg_distribution(
            pos_mask, neg_mask, total_count=total_samples, positive_ratio=pos_ratio)

        # calculate binary cross entropy with logits
        cls_loss = F.binary_cross_entropy_with_logits(
            preds[pos_mask | neg_mask], t_objness[pos_mask | neg_mask], reduction='sum')

        # calculate smooth l1 loss for bbox regression
        reg_loss = F.smooth_l1_loss(regs[pos_mask][:,0], t_regs[pos_mask][:,0], reduction='sum') +\
            F.smooth_l1_loss(regs[pos_mask][:,1], t_regs[pos_mask][:,1], reduction='sum') +\
                F.smooth_l1_loss(regs[pos_mask][:,2], t_regs[pos_mask][:,2], reduction='sum') +\
                    F.smooth_l1_loss(regs[pos_mask][:,3], t_regs[pos_mask][:,3], reduction='sum')

        return {
            'loss': (cls_loss)/Ncls +(w_lamda*reg_loss)/Nreg,
            'cls_loss':cls_loss.detach().item(),
            'reg_loss':reg_loss.detach().item()
        }

    def test_step(self, batch, targets):
        # preds: bs x nA x fmap_h x fmap_w x 2
        # regs: bs x nA x fmap_h x fmap_w x 4

        preds,regs = self.forward(batch)
        # TODO handle batch

        return self.detection_layer._inference_postprocess(preds,regs,targets[0]['img_dims'].to(preds.device))

if __name__ == "__main__":
    import torchvision.models as models
    import sys
    from cv2 import cv2

    img_path = sys.argv[1]

    batch,img = load_data(img_path)

    vgg = models.vgg16(pretrained=True).features[:-1]
    vgg.eval()

    img_dims = img.shape[:2]

    rpn = RPN()
    rpn.eval()

    with torch.no_grad():
        fmap = vgg(batch)
        det_results = rpn(fmap, img_dims=img_dims)

    for x1,y1,x2,y2,sc in det_results[0].long():
        o_img = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.imshow("",o_img)
        cv2.waitKey(0)