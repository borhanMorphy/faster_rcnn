import torch
from .box import jaccard_vectorized,xyxy2cxcywh
from typing import Tuple,List,Dict
import numpy as np
from cv2 import cv2

def build_targets(preds:torch.Tensor, regs:torch.Tensor, default_boxes:torch.Tensor, gt_boxes:torch.Tensor, gt_classes:torch.Tensor,
        negative_iou_threshold:float=0.3, positive_iou_threshold:float=0.7, ignore_indexes:torch.Tensor=None):
    """[summary]

    Args:
        preds (torch.Tensor): nA x Gy x Gx x 2
        regs (torch.Tensor): nA x Gy x Gx x 4
        default_boxes (torch.Tensor): (nA * Gy * Gx) x 4
        negative_iou_threshold (float, optional): [description]. Defaults to 0.3.
        positive_iou_threshold (float, optional): [description]. Defaults to 0.7.
        ignore_indexes (torch.Tensor, optional): X,4 Defaults to None.

    Returns:
        p_preds.shape(Np,2) == gt_preds.shape(Np,2) # as {FG,BG}
        p_regs.shape(Nr,4) == gt_regs.shape(Nr,4) # as (tx,ty,tw,th)
        (p_preds,gt_preds),(p_regs,gt_regs)
    """
    nA,Gy,Gx,_ = preds.shape

    p_pos_preds = preds.reshape(-1,2)[:, 1]
    p_neg_preds = preds.reshape(-1,2)[:, 0]
    gt_pos_preds = torch.zeros(*(nA*Gy*Gx,), dtype=preds.dtype, device=preds.device)
    gt_neg_preds = torch.zeros(*(nA*Gy*Gx,), dtype=preds.dtype, device=preds.device)

    p_regs = regs.reshape(-1,4)

    # ps: N = nA * Gy * Gx
    # N,4 | M,4 => N,M
    ious = jaccard_vectorized(default_boxes, gt_boxes)

    max_values,matched_gt_indexes = ious.max(dim=1)
    # max_values: N, {iou values between prediction and matched gt}
    # matched_gt_indexes: N, {gt index that matched with prediction}

    positives_mask = max_values > positive_iou_threshold # N, boolean mask
    negatives_mask = max_values < negative_iou_threshold # N, boolean mask

    for j in range(gt_boxes.size(0)):
        for k in ious[:,j].argsort(descending=True):
            if ignore_indexes is not None and k in ignore_indexes: continue
            positives_mask[k] = True
            negatives_mask[k] = False
            break

    if ignore_indexes is not None:
        positives_mask[ignore_indexes] = False
        negatives_mask[ignore_indexes] = False

    gt_pos_preds[positives_mask] = 1.
    gt_neg_preds[negatives_mask] = 1.

    # resample positives and negatives as 128:128
    positives, = torch.where(positives_mask)
    negatives, = torch.where(negatives_mask)
    positive_count = min(positives.size(0),128)
    negative_count = 256-positive_count

    picked_positives = np.random.choice(positives.cpu().numpy(), size=positive_count, replace=False)
    picked_negatives = np.random.choice(negatives.cpu().numpy(), size=negative_count, replace=False)
    ##################################################

    p_regs = p_regs[picked_positives]
    # TODO convert gt boxes to gt offsets
    gt_regs = gt_boxes[matched_gt_indexes][picked_positives]
    gt_regs = xyxy2offsets(gt_regs, default_boxes[picked_positives])

    targets = []
    targets.append( (p_pos_preds[picked_positives], gt_pos_preds[picked_positives]) )
    targets.append( (p_neg_preds[picked_negatives], gt_neg_preds[picked_negatives]) )
    targets.append( (p_regs,gt_regs) )
    return targets


def xyxy2offsets(boxes, anchors):
    """Convert boxes to offsets

    Args:
        boxes ([type]): [description]
    """
    """
    if img is not None:
        for box,anchor in zip(boxes,anchors):
            box = box.long().cpu().numpy()
            anchor = anchor.long().cpu().numpy()
            c_img = img.copy()
            c_img = cv2.rectangle(c_img, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
            c_img = cv2.rectangle(c_img, (anchor[0],anchor[1]), (anchor[2],anchor[3]), (0,0,255), 2)
            print(box,anchor)
            cv2.imshow("",c_img)
            cv2.waitKey(0)
    """
    c_boxes = xyxy2cxcywh(boxes)
    c_anchors = xyxy2cxcywh(anchors)

    tx = (c_boxes[:, 0] - c_anchors[:, 0]) / c_anchors[:, 2]
    ty = (c_boxes[:, 1] - c_anchors[:, 1]) / c_anchors[:, 3]
    tw = torch.log( c_boxes[:, 2] / c_anchors[:, 2] )
    th = torch.log( c_boxes[:, 3] / c_anchors[:, 3] )

    return torch.stack([tx,ty,tw,th], dim=-1)