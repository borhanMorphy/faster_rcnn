import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool,RoIPool
from torchvision.ops.boxes import batched_nms
from typing import List,Dict,Tuple
from utils import (
    apply_box_regressions,
    build_targets_v2,
    resample_pos_neg_distribution_v2,
    jaccard_vectorized,
    xyxy2offsets
)
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def draw_it(pboxes,tboxes):
    import numpy as np
    from cv2 import cv2
    blank = np.ones((1000,1000,3),dtype=np.uint8) * 255

    for (x11,y11,x12,y12),(x21,y21,x22,y22) in zip(pboxes.cpu().long().numpy(),tboxes.cpu().long().numpy()):
        t_blank = blank.copy()
        t_blank = cv2.rectangle(t_blank,(x11,y11),(x12,y12),(0,0,255),2)
        t_blank = cv2.rectangle(t_blank,(x21,y21),(x22,y22),(0,255,0),2)
        cv2.imshow("",t_blank)
        cv2.waitKey(0)

class FastRCNN(nn.Module):
    def __init__(self, num_classes:int, effective_stride:int,
            output_size:int, features:int, hidden_channels:int=1024, keep_n:int=100,
            num_of_samples:int=512, positive_ratio:float=.25,
            positive_iou_threshold:float=0.5, negative_iou_threshold:float=0.5,
            train_conf_threshold:float=.05, test_conf_threshold:float=.05):
        super(FastRCNN,self).__init__()
        self.num_classes = num_classes + 1 # adding background
        self.effective_stride = effective_stride
        self.roi_pool = RoIPool(output_size,1.0)
        self.hidden_unit = nn.Sequential(
            nn.Linear(
                in_features=output_size*output_size*features,
                out_features=hidden_channels, bias=True),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=hidden_channels,
                out_features=hidden_channels, bias=True),
            nn.Dropout(p=0.5))

        self.cls_unit = nn.Linear(hidden_channels, self.num_classes)
        self.reg_unit = nn.Linear(hidden_channels, self.num_classes*4)


        self._params = {
            'train':{
                'conf_threshold':train_conf_threshold
            },
            'test':{
                'conf_threshold':test_conf_threshold
            },
            'positive_iou_threshold': positive_iou_threshold,
            'negative_iou_threshold': negative_iou_threshold,
            'num_of_samples': num_of_samples,
            'positive_ratio': positive_ratio,
            'keep_n':keep_n
        }

    def get_params(self):
        return self._params['train'] if self.training else self._params['test']

    def forward(self, fmap:torch.Tensor, rois:List[torch.Tensor],
            targets:List[Dict[str,torch.Tensor]]=None):
        # ! assumed batch size is 1
        # TODO add batch size support
        bs = fmap.size(0)
        rois = rois[0] # ! bs 1 assumed

        params = self.get_params()

        if targets is not None:
            # select and assign rois
            rois,targets = self.select_training_samples(rois,targets[0])
        
        outputs = self.roi_pool(fmap, [rois]).flatten(start_dim=1) # N,C,H,W | [(M,4), ...] => K,(C*output_size[0]*output_size[1])
        outputs = self.hidden_unit(outputs) # K,hiddin_channels
        preds = self.cls_unit(outputs) # K,num_classes+1
        regs = self.reg_unit(outputs)  # K,(num_classes+1)*4

        # K,(num_classes+1)*4 => K,num_classes+1,4
        regs = regs.reshape(-1,self.num_classes,4)

        cls_scores,cls_ids = F.softmax(preds, dim=1).max(dim=1)

        # K,num_classes+1,4 => K,4
        offsets = regs.clone().gather(1, cls_ids.view(-1,1).repeat(1,4).unsqueeze(1)).squeeze(1)
        
        ignore_bg = cls_ids != 0

        # cls_ids: K,
        # cls_scores: K,
        boxes = apply_box_regressions(offsets.unsqueeze(0), rois).squeeze(0) # ! assumed bs is 1
        pick = cls_scores >= params['conf_threshold']
        boxes = boxes[pick & ignore_bg]
        cls_ids = cls_ids[pick & ignore_bg].float().unsqueeze(-1)
        cls_scores = cls_scores[pick & ignore_bg].unsqueeze(-1)


        #pick = batched_nms(boxes,cls_scores,cls_ids,iou_threshold)
        #boxes = boxes[pick]
        #cls_ids = cls_ids[pick]
        #cls_scores = cls_scores[pick]
        # T,4 | T,1 | T,1 => T,6
        dets = torch.cat([boxes,cls_scores,cls_ids], dim=-1)

        if targets is not None:
            # * derivetive of rois is not needed
            losses = self.compute_loss(preds,regs,rois,targets)
            return dets,losses
        return dets

    def select_training_samples(self, rois:torch.Tensor, targets:Dict[str,torch.Tensor]):
        # assumed bs is 1
        """
            rois: N,4
            gt_boxes: M,4
            gt_labels: M,

            rois: S,4
            target_regs: S,4
            target_labels: S,
        """
        
        gt_boxes = targets['boxes'].to(rois.device)
        gt_labels = targets['labels'].to(rois.device)
        N = rois.size(0)

        fg_mask = torch.zeros(*(N,), dtype=torch.bool, device=rois.device)
        bg_mask = torch.zeros(*(N,), dtype=torch.bool, device=rois.device)

        target_regs = torch.zeros(*(N,4), dtype=rois.dtype, device=rois.device) - 1
        target_labels = torch.zeros(*(N,), dtype=gt_labels.dtype, device=rois.device) - 1


        # find positive candidates
        ious = jaccard_vectorized(rois, gt_boxes) # => N,4 | M,4 => N,M
        max_iou_values,max_iou_ids = ious.max(dim=-1)

        # set fg if iou is higher than threshold given
        pmask = max_iou_values >= self._params['positive_iou_threshold']
        fg_mask[pmask] = True
        for gt_idx in range(gt_boxes.size(0)):
            target_labels[ pmask & (max_iou_ids == gt_idx) ] = gt_labels[gt_idx]
            target_regs[ pmask & (max_iou_ids == gt_idx) ] = gt_boxes[gt_idx]

        # set bg if iou is lower than threshold given
        nmask = max_iou_values < self._params['negative_iou_threshold']
        bg_mask[nmask] = True
        target_labels[bg_mask] = 0 # set label as background


        selected_pos,selected_neg = resample_pos_neg_distribution_v2(
            fg_mask, bg_mask, total_count=self._params['num_of_samples'],
            positive_ratio=self._params['positive_ratio'])

        fg_mask[~selected_pos] = False
        bg_mask[~selected_neg] = False

        target_labels = target_labels[ fg_mask | bg_mask ]
        target_regs = target_regs[ fg_mask | bg_mask ]
        rois = rois[ fg_mask | bg_mask ]

        #! warning this might cause error since its calculated with bg indexes too
        target_regs = xyxy2offsets(target_regs,rois)

        targets = torch.cat([target_regs,target_labels.float().unsqueeze(-1)], dim=-1)

        return rois,targets

    def compute_loss(self, preds:torch.Tensor, regs:torch.Tensor, rois:torch.Tensor,
            targets:torch.Tensor) -> Dict[str,torch.Tensor]:
        S = preds.size(0)
        nc = self.num_classes
        fg_mask = targets[:,4] != 0
        cls_mask = torch.zeros((S,nc), dtype=torch.bool, device=preds.device)
        for i,cls_idx in enumerate(targets[:,4].long()):
            cls_mask[i][cls_idx] = True

        #print(f"total positives: {pos_mask[pos_mask].size(0)} total negatives: {neg_mask[neg_mask].size(0)}")
        # preds[selected_pos | selected_neg] (num_of_samples,21) as float
        # t_preds[pos_mask | neg_mask] (num_of_samples,) as long
        # calculate cross entropy
        cls_loss = F.cross_entropy(preds,targets[:,4].long())

        # calculate smooth l1 loss for bbox regression
        if fg_mask[fg_mask].size(0) == 0:
            reg_loss = torch.tensor(0., requires_grad=True, dtype=regs.dtype, device=regs.device)
        else:
            reg_loss = F.smooth_l1_loss(regs[cls_mask][fg_mask], targets[fg_mask,:4], reduction='sum') / self.num_classes

        if torch.isnan(reg_loss):
            print("***********************************debug***********************************")
            print(targets,rois)
            print("***********************************end***********************************")
        
        return {'cls_loss':cls_loss,'reg_loss':reg_loss}