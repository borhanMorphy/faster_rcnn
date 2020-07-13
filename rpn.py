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
    get_ignore_boxes
)

from utils.train import (
    build_targets
)
from utils.data import load_data


class DetectionLayer(nn.Module):
    def __init__(self, effective_stride:int=16,
            anchor_scales:List=[0.5,1,2], anchor_ratios:List=[0.5,1,2],
            iou_threshold:float=0.7, keep_n:int=300):
        super(DetectionLayer,self).__init__()

        self.effective_stride = effective_stride
        self.iou_threshold = iou_threshold
        self.keep_n = keep_n

        self.anchors = generate_anchors(self.effective_stride,
            ratios=anchor_ratios, scales=anchor_scales)

        self._fw = 62
        self._fh = 37
        self.default_boxes = generate_default_boxes(
            self.anchors, (self._fh, self._fw), self.effective_stride)

    def forward(self, preds:torch.Tensor, regs:torch.Tensor) -> List[torch.Tensor]:

        bs,_,fh,fw = preds.shape
        # bs x num_anchors*2 x fmap_h x fmap_w => bs x 2 x nA x fmap_h x fmap_w => bs x nA x fmap_h x fmap_w x 2
        preds = preds.reshape(bs,2,-1,fh,fw).permute(0,2,3,4,1).contiguous()

        # bs x num_anchors*4 x fmap_h x fmap_w => bs x 4 x nA x fmap_h x fmap_w => bs x nA x fmap_h x fmap_w x 4
        regs = regs.reshape(bs,4,-1,fh,fw).permute(0,2,3,4,1).contiguous()

        if fw != self._fw or fh != self._fh:
            # re-generate default boxes if feature map size is changed
            self._fw = fw
            self._fh = fh
            self.default_boxes = generate_default_boxes( # nA x (fmap_h * fmap_w) x 4 as xmin ymin xmax ymax
                self.anchors, (self._fh, self._fw),
                self.effective_stride, device=preds.device)

        if preds.device != self.default_boxes.device:
            self.default_boxes = self.default_boxes.to(preds.device)

        return preds,regs

    def _inference_postprocess(self, preds:torch.Tensor, regs:torch.Tensor, img_dims:torch.Tensor):
        """
        Arguments:
            preds: bs x nA x fmap_h x fmap_w x 2
            regs:  bs x nA x fmap_h x fmap_w x 4
            img_dims: torch.tensor(height,width)
        """
        bs = preds.size(0)

        # bs x nA x fmap_h x fmap_w x 4 => bs x N
        scores = F.softmax(preds.reshape(bs,-1,2), dim=-1)[:,:,1]

        # bs x nA x fmap_h x fmap_w x 4 => bs x N x 4
        boxes = apply_box_regressions(self.default_boxes, regs)

        boxes = clip_boxes(boxes, img_dims)

        det_results = []
        for i in range(bs):
            pick = nms(boxes[i], scores[i], self.iou_threshold)
            bb = boxes[i,pick,:]
            sc = scores[i,pick]
            sort = sc.argsort(descending=True)
            bboxes = torch.cat([bb[sort][:self.keep_n], sc[sort][:self.keep_n].unsqueeze(-1)],dim=1)
            det_results.append(bboxes)

        return det_results


class RPN(nn.Module):
    """Input features represents vgg16 backbone and n=3 is set because of the `faster rcnn` parper

    Number of anchors selected from paper, where num_anchors: num_scales * num_ratios
    """
    def __init__(self, features:int=512, n:int=3, effective_stride:int=16,
            anchor_scales:List=[0.5,1,2], anchor_ratios:List=[0.5,1,2]):
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
            in_channels=features, out_channels=num_anchors*2,
            kernel_size=1, stride=1, padding=0)

        self.reg_conv_layer = nn.Conv2d(
            in_channels=features, out_channels=num_anchors*4,
            kernel_size=1, stride=1, padding=0)

        self.detection_layer = DetectionLayer(effective_stride=effective_stride,
            anchor_ratios=anchor_ratios, anchor_scales=anchor_scales)

        self.debug = True

    def forward(self, fmap:torch.Tensor):
        output = self.base_conv_layer(fmap)

        preds = self.cls_conv_layer(output)
        regs = self.reg_conv_layer(output)

        return self.detection_layer(preds,regs)

    def training_step(self, batch, targets, imgs:List=None):
        # preds: bs x nA x fmap_h x fmap_w x 2
        # regs: bs x nA x fmap_h x fmap_w x 4
        preds,regs = self.forward(batch)
        bs = preds.size(0)

        for i in range(bs):

            ignore_indexes = get_ignore_boxes(self.detection_layer.default_boxes, img_dims=targets[i]['img_dims'])
            (p_preds,gt_preds),(p_regs,gt_regs) = build_targets(preds[i], regs[i],
                self.detection_layer.default_boxes,
                targets[i], ignore_indexes=ignore_indexes, img=imgs[i])

            """
            if self.debug:
                p_boxes = self.detection_layer.default_boxes.reshape(nA,fmap_h,fmap_w,4)[p_mask,:]
                # c,h,w => h,w,c
                img = imgs[i].copy()

                for x1,y1,x2,y2 in targets[i]['boxes'].cpu().long().numpy().tolist():
                    img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

                for x1,y1,x2,y2 in p_boxes.cpu().long().numpy().tolist():
                    c_img = cv2.rectangle(img.copy(),(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.imshow("",c_img)
                    cv2.waitKey(0)
            """

            # TODO calculate loss functions
            cls_loss = F.binary_cross_entropy_with_logits(p_preds[i][p_mask], t_preds[i][p_mask])


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