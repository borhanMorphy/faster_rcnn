import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool,RoIPool
from torchvision.ops.boxes import batched_nms
from typing import List,Dict,Tuple
from utils import (
    apply_box_regressions,
    build_targets_v2,
    resample_pos_neg_distribution_v2
)

class FastRCNN(nn.Module):
    def __init__(self, num_classes:int, effective_stride:int,
            output_size:int, features:int, hidden_channels:int=1024,
            num_of_samples:int=512, positive_ratio:float=0.25):
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
            'num_of_samples': num_of_samples,
            'positive_ratio': positive_ratio
        }

    def forward(self, fmap:torch.Tensor, rois:List[torch.Tensor],
            targets:List[Dict[str,torch.Tensor]]=None,
            conf_threshold:float=0.01, iou_threshold:float=0.7):
        # ! assumed batch size is 1
        # TODO add batch size support
        bs = fmap.size(0)
        rois = rois[0] # ! bs 1 assumed
        
        outputs = self.roi_pool(fmap, [rois]).flatten(start_dim=1) # N,C,H,W | [(M,4), ...] => K,(C*output_size[0]*output_size[1])
        outputs = self.hidden_unit(outputs) # K,hiddin_channels
        preds = self.cls_unit(outputs) # K,num_classes+1
        regs = self.reg_unit(outputs)  # K,(num_classes+1)*4

        # K,(num_classes+1)*4 => K,num_classes+1,4
        regs = regs.reshape(-1,self.num_classes,4)

        cls_scores,cls_ids = F.softmax(preds, dim=1).max(dim=1)
        # K,num_classes+1,4 => K,4
        offsets = regs.clone().gather(1, cls_ids.view(-1,1).repeat(1,4).unsqueeze(1)).squeeze(1)

        # cls_ids: K,
        # cls_scores: K,
        boxes = apply_box_regressions(offsets.unsqueeze(0), rois).squeeze(0) # ! assumed bs is 1
        pick = cls_scores >= conf_threshold
        boxes = boxes[pick]
        cls_ids = cls_ids[pick]
        cls_scores = cls_scores[pick]

        pick = batched_nms(boxes,cls_scores,cls_ids,iou_threshold)
        boxes = boxes[pick]
        cls_ids = cls_ids[pick].float().unsqueeze(-1)
        cls_scores = cls_scores[pick].unsqueeze(-1)
        # T,4 | T,1 | T,1 => T,6
        dets = torch.cat([boxes,cls_ids,cls_scores], dim=-1)

        if targets is not None:
            # * derivetive of rois is not needed
            losses = self.compute_loss(preds,regs,rois[:,:4].detach(),targets[0]) # ! assumed bs is 1
            return dets,losses

        return dets


    def compute_loss(self, preds:torch.Tensor, regs:torch.Tensor, rois:List[torch.Tensor],
            targets:List[Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:

        (pos_mask,neg_mask),(t_preds,t_regs) = build_targets_v2(preds,regs,rois,targets)

        # resample positive and negatives to have ratio of 1:1 (pad with negatives if needed)
        selected_pos,selected_neg = resample_pos_neg_distribution_v2(
            pos_mask, neg_mask, total_count=self._params['num_of_samples'],
            positive_ratio=self._params['positive_ratio'])

        pos_mask[~selected_pos] = False
        neg_mask[~selected_neg] = False

        # calculate binary cross entropy with logits
        cls_loss = F.cross_entropy(preds[selected_pos | selected_neg], t_preds[pos_mask | neg_mask].long())

        # calculate smooth l1 loss for bbox regression
        reg_loss = F.smooth_l1_loss(regs[pos_mask][:,0], t_regs[pos_mask][:,0]) +\
            F.smooth_l1_loss(regs[pos_mask][:,1], t_regs[pos_mask][:,1]) +\
                F.smooth_l1_loss(regs[pos_mask][:,2], t_regs[pos_mask][:,2]) +\
                    F.smooth_l1_loss(regs[pos_mask][:,3], t_regs[pos_mask][:,3])

        return {'cls_loss':cls_loss,'reg_loss':reg_loss}