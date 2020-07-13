import torch
from .box import jaccard_vectorized
from typing import Tuple,List
import numpy as np

def assign_targets(preds:torch.Tensor, default_boxes:torch.Tensor, target_boxes:torch.Tensor,
        negative_iou_threshold:float=0.3, positive_iou_threshold:float=0.7, ignore_indexes:List=None):
    # TODO check ignore indexes
    """Assigns negative to positives

    Arguments:
        preds: nA x grid_y x grid_x x 2
        default_boxes: (nA*fmap_h*fmap_w),4

    Returns:
        positive_mask: nA x grid_y x grid_x
    """

    num_anchors,grid_y,grid_x,_ = preds.shape
    device = preds.device
    positive_mask = torch.zeros(*(num_anchors,grid_y,grid_x), dtype=torch.bool, device=device)
    negative_mask = torch.zeros(*(num_anchors,grid_y,grid_x), dtype=torch.bool, device=device)

    # N,4 | M,4 => N,M
    ious = jaccard_vectorized(default_boxes, target_boxes)

    pos_proposals,pos_gts = torch.where(ious > positive_iou_threshold)
    neg_proposals,_ = torch.where(ious < negative_iou_threshold)

    positive_mask = positive_mask.reshape(-1)
    positive_mask[pos_proposals] = True

    negative_mask = negative_mask.reshape(-1)
    negative_mask[neg_proposals] = True

    # add best matches
    for j in range(target_boxes.size(0)):
        for k in ious[:,j].argsort(descending=True):
            if ignore_indexes is not None and k in ignore_indexes:
                continue
            positive_mask[k] = True

    # extract negative indexes
    if ignore_indexes is not None:
        positive_mask[ignore_indexes] = False
        negative_mask[ignore_indexes] = False
    
    positive_mask = positive_mask.reshape(num_anchors,grid_y,grid_x)
    negative_mask = negative_mask.reshape(num_anchors,grid_y,grid_x)

    return positive_mask,negative_mask

def sample_positive_and_negatives(positives, negatives,
        positive_count:int=128, total_count:int=256):
    """
        positives: nA x fmap_h x fmap_w
        negatives: nA x fmap_h x fmap_w
    """
    nA,fh,fw = positives.shape

    s_positives = torch.zeros(*(positives.reshape(-1).size(0),), dtype=positives.dtype, device=positives.device)
    s_negatives = torch.zeros(*(negatives.reshape(-1).size(0),), dtype=negatives.dtype, device=negatives.device)

    pos_population, = torch.where(positives.reshape(-1))
    positive_count = min(pos_population.size(0),positive_count)
    pick_positives = np.random.choice(pos_population.cpu().numpy(), size=positive_count, replace=False)
    s_positives[pick_positives] = True

    neg_population, = torch.where(negatives.reshape(-1))
    negative_count = total_count-positive_count
    pick_negatives = np.random.choice(neg_population.cpu().numpy(), size=negative_count, replace=False)
    s_negatives[pick_negatives] = True

    s_positives = s_positives.reshape(nA,fh,fw)
    s_negatives = s_negatives.reshape(nA,fh,fw)

    return s_positives,s_negatives