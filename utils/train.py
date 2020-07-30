import torch
from .box import (
    jaccard_vectorized,
    _vector2anchor,
    _anchor2vector
)
from typing import Tuple,List,Dict
import numpy as np
from cv2 import cv2

def build_targets_v2(preds:torch.Tensor, regs:torch.Tensor,
        default_boxes:torch.Tensor, targets:Dict[str,torch.Tensor],
        positive_iou_threshold:float=0.5):
    """[summary]

    Args:
        preds (torch.Tensor): K x num_classes # including background
        regs (torch.Tensor): K x num_classes x 4
        default_boxes (torch.Tensor): K x 4
        targets (Dict[str,torch.Tensor]]):{
            'boxes':torch.tensor(T,4) as xmin,ymin,xmax,ymax
            'labels':torch.tensor(T,) as label
        }
    Returns:
        Tuple
            Tuple
                positive_mask (torch.Tensor): K x nc
                negative_mask (torch.Tensor): K x nc
            Tuple
                target_preds (torch.Tensor): K x nc
                target_regs (torch.Tensor): K x nc x 4
    """
    K,nc = preds.shape
    positive_mask = torch.zeros(*(K,nc,), dtype=torch.bool, device=preds.device)
    negative_mask = torch.zeros(*(K,nc,), dtype=torch.bool, device=preds.device)

    target_preds = torch.zeros(*(K,nc), dtype=preds.dtype, device=preds.device)
    target_regs = torch.zeros(*(K,nc,4), dtype=regs.dtype, device=regs.device)

    gt_boxes = targets['boxes'].to(regs.device)
    gt_labels = targets['labels'].float().to(preds.device)

    # find positive candidates
    ious = jaccard_vectorized(default_boxes, gt_boxes) # => K,4 | T,4 => K,T
    max_iou_values,max_iou_ids = ious.max(dim=-1)

    # set True best matched gt with anchors
    best_matches = ious.argmax(dim=0)
    for j in range(best_matches.size(0)):
        cls_id = gt_labels[j].long()
        target_preds[best_matches[j]] = cls_id
        target_regs[best_matches[j]] = gt_boxes[j]

    positive_mask[best_matches] = True

    # set True if iou is higher than threshold given
    pmask = max_iou_values >= positive_iou_threshold
    for gt_idx in range(gt_boxes.size(0)):
        target_preds[ pmask & (max_iou_ids == gt_idx) ] = gt_labels[gt_idx]
        # TODO convert to offsets
        target_regs[ pmask & (max_iou_ids == gt_idx) ] = gt_boxes[gt_idx]

    positive_mask[max_iou_values >= positive_iou_threshold] = True
    negative_mask[~positive_mask] = True 
    negative_mask[:,1:] = False

    # TODO vectorize
    mask = torch.ones(nc, dtype=torch.bool, device=preds.device)
    for i in range(positive_mask.size(0)):
        cls_idx = target_preds[i][0].long()
        mask[cls_idx] = False
        positive_mask[i,mask] = False
        target_preds[i,mask] = .0
        mask[cls_idx] = True
    
    # convert regs, boxes => offsets
    k_indexes,_ = torch.where(positive_mask)
    target_regs[positive_mask] = xyxy2offsets(target_regs[positive_mask], default_boxes[k_indexes])
    return (positive_mask,negative_mask),(target_preds,target_regs)


def build_targets(preds:torch.Tensor, regs:torch.Tensor, default_boxes:torch.Tensor,
        ignore_mask:torch.Tensor, targets:List, negative_iou_threshold:float=0.3,
        positive_iou_threshold:float=0.7, imgs=None):
    # ! ignore boxes ignored after assigment, this can cause best matches to be ignored
    """builds traning targets

    Args:
        preds (torch.Tensor): bs x fh x fw x nA
        regs (torch.Tensor): bs x fh x fw x nA x 4
        default_boxes (torch.Tensor): fh x fw x nA x 4
        ignore_mask(torch.Tensor): fh x fw x nA
        targets:List[Dict]
            'boxes': torch.FloatTensor as M,4
            'labels': torch.LongTensor as M,
        negative_iou_threshold (float, optional): [description]. Defaults to 0.3.
        positive_iou_threshold (float, optional): [description]. Defaults to 0.7.

    Returns:
        Tuple
            Tuple
                positive_mask (torch.Tensor): bs x fh x fw x nA
                negative_mask (torch.Tensor): bs x fh x fw x nA
            Tuple
                target_objness (torch.Tensor): bs x fh x fw x nA
                target_preds (torch.Tensor): bs x fh x fw x nA
                target_regs (torch.Tensor): bs x fh x fw x nA x 4
    """
    bs,fh,fw,nA = preds.shape
    device = preds.device

    positive_mask = torch.zeros(*(bs,fh,fw,nA), dtype=torch.bool, device=device)
    negative_mask = torch.zeros(*(bs,fh,fw,nA), dtype=torch.bool, device=device)

    target_objness = torch.zeros(*(bs,fh,fw,nA), dtype=preds.dtype, device=device) - 1 # set -1 as ignore
    target_preds = torch.zeros(*(bs,fh,fw,nA), dtype=preds.dtype, device=device)
    target_regs = torch.zeros(*(bs,fh,fw,nA,4), dtype=regs.dtype, device=device)

    for i in range(bs):
        gt_boxes = targets[i]['boxes'].to(regs.dtype)
        gt_labels = targets[i]['labels'].to(preds.dtype)
        if gt_boxes.device != device: gt_boxes = gt_boxes.to(device)
        if gt_labels.device != device: gt_labels = gt_labels.to(device)

        if gt_boxes.size(0) == 0:
            raise AssertionError("ground truths cannot be empy")

        t_pred = target_preds[i].reshape(-1)
        t_reg = target_regs[i].reshape(-1,4)
        pos = positive_mask[i].reshape(-1)
        neg = negative_mask[i].reshape(-1)

        # find positive candidates
        ious = jaccard_vectorized(default_boxes.reshape(-1,4), gt_boxes) # => N,4 | M,4 => N,M
        max_iou_values,max_iou_ids = ious.max(dim=-1)

        # set True best matched gt with anchors
        best_matches = ious.argmax(dim=0)
        for j in range(best_matches.size(0)):
            t_pred[ best_matches[j] ] = gt_labels[j]
            t_reg[ best_matches[j] ] = gt_boxes[j]

        pos[best_matches] = True
        pos = pos.reshape(fh,fw,nA)
        # set True if iou is higher than threshold given
        pmask = max_iou_values >= positive_iou_threshold
        for gt_idx in range(gt_boxes.size(0)):
            t_pred[ pmask & (max_iou_ids == gt_idx) ] = gt_labels[gt_idx]
            t_reg[ pmask & (max_iou_ids == gt_idx) ] = gt_boxes[gt_idx]

        pos[(max_iou_values >= positive_iou_threshold).reshape(fh,fw,nA)] = True  # fh,fw,nA

        # find negative candidates
        neg[max_iou_values <= negative_iou_threshold] = True  # fh,fw,nA
        # extract best matches
        neg[best_matches] = False
        neg = neg.reshape(fh,fw,nA)

        # extract igores
        pos[ignore_mask] = False
        neg[ignore_mask] = False

        # reshape
        t_pred = t_pred.reshape(fh,fw,nA)
        t_reg = t_reg.reshape(fh,fw,nA,4)

        # set objectness
        # -1 is ignore
        target_objness[i][pos] = 1.
        target_objness[i][neg] = 0.

        # convert regs, boxes => offsets
        t_reg[pos] = xyxy2offsets(t_reg[pos],default_boxes[pos])

        """
        ##* DEBUG
        print("total proposals: ", default_boxes.reshape(-1).shape)
        print("target pred: ",t_pred[pos].shape)
        print("target reg: ",t_reg[pos].shape)
        print("target objness: ",target_objness[i][pos].shape)
        print("negatives: ",neg[neg].shape)
        print("total gt: ",gt_boxes.size(0))
        print("total positives: ",pos[pos].reshape(-1).size(0))
        print("_"*50)
        draw_it(imgs[i],default_boxes[pos],default_boxes[neg],gt_boxes,(t_pred[pos],t_reg[pos]))
        ##* END
        """

        positive_mask[i] = pos
        negative_mask[i] = neg
        target_preds[i] = t_pred
        target_regs[i] = t_reg

    return (positive_mask,negative_mask),(target_objness,target_preds,target_regs)

def draw_it(img,p_boxes,n_boxes,gt_boxes,targets):
    t_pred,t_reg = targets
    for pbox,t_p,t_r in zip(p_boxes,t_pred,t_reg):
        print(f"box: {pbox.cpu().numpy().tolist()} assigned to cls: {t_p.cpu().numpy().tolist()} and reg:{t_r.cpu().numpy().tolist()}")


    pb = p_boxes.cpu().long().numpy()
    nb = n_boxes.cpu().long().numpy()
    gtb = gt_boxes.cpu().long().numpy()

    c_img = img.copy()
    for x1,y1,x2,y2 in gtb:
        c_img = cv2.rectangle(c_img, (x1,y1),(x2,y2), (0,255,0), 2)
    cv2.imshow("",c_img)
    cv2.waitKey(0)

    c_img = img.copy()
    for x1,y1,x2,y2 in pb:
        c_img = cv2.rectangle(c_img, (x1,y1),(x2,y2), (0,0,255), 2)
    cv2.imshow("",c_img)
    cv2.waitKey(0)

    c_img = img.copy()
    for x1,y1,x2,y2 in nb[:pb.shape[0]]:
        c_img = cv2.rectangle(c_img, (x1,y1),(x2,y2), (255,0,0), 2)
    cv2.imshow("",c_img)
    cv2.waitKey(0)

def xyxy2offsets(boxes:torch.Tensor, anchors:torch.Tensor):
    """Convert boxes to offsets

    Args:
        boxes (torch.Tensor): N,4 as xmin,ymin,xmax,ymax
        anchors (torch.Tensor): N,4 as xmin,ymin,xmax,ymax
    Returns:
        offsets (torch.Tensor): N,4 as tx,ty,tw,th
    """
    assert boxes.size(0) == anchors.size(0),"boxes and anchors does not match in dim 0"
    assert boxes.size(1) == anchors.size(1),"boxes and anchors does not match in dim 1"
    eps = 1e-8
    a_x_ctr,a_y_ctr,a_w,a_h = _anchor2vector(anchors)
    b_x_ctr,b_y_ctr,b_w,b_h = _anchor2vector(boxes)

    tx = (b_x_ctr-a_x_ctr) / a_w
    ty = (b_y_ctr-a_y_ctr) / a_h
    if (b_w <= 0).any() or (a_w <= 0).any() : print("danger!!! ",b_w,a_w)
    if (b_h <= 0).any() or (a_h <= 0).any() : print("danger!!! ",b_h,a_h)

    tw = torch.log(b_w/a_w + eps)
    th = torch.log(b_h/a_h + eps)

    return torch.stack([tx,ty,tw,th], dim=-1)

def offsets2xyxy(offsets:torch.Tensor, anchors:torch.Tensor):
    """Convert offsets to boxes

    Args:
        offsets (torch.Tensor): N,4 as tx,ty,tw,th
        anchors (torch.Tensor): N,4 as xmin,ymin,xmax,ymax

    Returns:
        boxes (torch.Tensor): N,4 as xmin,ymin,xmax,ymax
    """
    a_x_ctr,a_y_ctr,a_w,a_h = _anchor2vector(anchors)
    tx = offsets[:, 0]
    ty = offsets[:, 1]
    tw = offsets[:, 2]
    th = offsets[:, 3]

    b_x_ctr = tx*a_w + a_x_ctr
    b_y_ctr = ty*a_h + a_y_ctr
    b_w = torch.exp(tw)*a_w
    b_h = torch.exp(th)*a_h

    return _vector2anchor(b_x_ctr,b_y_ctr,b_w,b_h)

def resample_pos_neg_distribution_v2(positive_mask:torch.Tensor, negative_mask:torch.Tensor,
        total_count:int=512, positive_ratio:float=.25):
    # !
    # TODO bs will cause error if bs > 1, fix it later
    K = positive_mask.size(0)

    n_positive_mask = torch.zeros(*(K,), dtype=torch.bool, device=positive_mask.device)
    n_negative_mask = torch.zeros(*(K,), dtype=torch.bool, device=negative_mask.device)

    # get indexes
    positives,*_ = torch.where(positive_mask)
    negatives,*_ = torch.where(negative_mask)

    pos_count = int(total_count*positive_ratio)
    total_count = min(total_count, positives.size(0) + negatives.size(0))
    
    positive_count = min(positives.size(0),pos_count)
    negative_count = total_count-positive_count
    negative_count = min(negative_count,negatives.size(0))

    picked_positives = np.random.choice(positives.cpu().numpy(), size=positive_count, replace=False)
    picked_negatives = np.random.choice(negatives.cpu().numpy(), size=negative_count, replace=False)

    n_positive_mask[picked_positives] = True
    n_negative_mask[picked_negatives] = True

    return n_positive_mask,n_negative_mask

def resample_pos_neg_distribution(positive_mask:torch.Tensor, negative_mask:torch.Tensor,
        total_count:int=256, positive_ratio:float=.5):
    # !
    # TODO bs will cause error if bs > 1, fix it later
    bs,fh,fw,nA = positive_mask.shape

    n_positive_mask = torch.zeros(*(bs*fh*fw*nA,), dtype=torch.bool, device=positive_mask.device)
    n_negative_mask = torch.zeros(*(bs*fh*fw*nA,), dtype=torch.bool, device=negative_mask.device)

    # get indexes
    positives, = torch.where(positive_mask.reshape(-1))
    negatives, = torch.where(negative_mask.reshape(-1))
    total_count = min(positives.size(0)+negatives.size(0),total_count)
    pos_count = int(total_count*positive_ratio)
    positive_count = min(positives.size(0),pos_count)
    negative_count = total_count-positive_count
    negative_count = min(negative_count,negatives.size(0))

    picked_positives = np.random.choice(positives.cpu().numpy(), size=positive_count, replace=False)
    picked_negatives = np.random.choice(negatives.cpu().numpy(), size=negative_count, replace=False)

    n_positive_mask[picked_positives] = True
    n_negative_mask[picked_negatives] = True

    n_positive_mask = n_positive_mask.reshape(bs,fh,fw,nA)
    n_negative_mask = n_negative_mask.reshape(bs,fh,fw,nA)

    return n_positive_mask,n_negative_mask

