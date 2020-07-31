import torch
from typing import Tuple,List,Dict
import numpy as np
from cv2 import cv2
from torchvision.ops import boxes as box_ops
from utils import boxes2offsets

def build_targets(batched_anchors:List[torch.Tensor],
        targets:List[Dict[str,torch.Tensor]],
        pos_iou_tresh:float, neg_iou_tresh:float,
        add_best_matches:bool=False) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """
    Params:

    Returns:
        matches torch.Tensor: Ntotal,
        target_objectness torch.Tensor : Ntotal, 
        target_labels torch.Tensor : Ntotal, 
        target_offsets torch.Tensor : Ntotal, 4
    """
    batch_matches:List[torch.Tensor] = []
    batch_target_objectness:List[torch.Tensor] = []
    batch_target_labels:List[torch.Tensor] = []
    batch_target_offsets:List[torch.Tensor] = []

    for anchors,targets_per_image in zip(batched_anchors,targets):
        device = anchors.device
        dtype = anchors.dtype

        gt_boxes = targets_per_image['boxes'].to(device, dtype)
        gt_labels = targets_per_image['labels'].to(device, torch.int64)

        N = anchors.size(0)
        M = gt_boxes.size(0)
        # -1: negative match | 0: ignore | 1: positive match
        matches = torch.zeros(N, dtype=dtype, device=device)
        target_objectness = torch.zeros(N, dtype=dtype, device=device)
        target_labels = torch.zeros(N, dtype=torch.int64, device=device)
        target_offsets = torch.zeros(N,4 , dtype=dtype, device=device)

        if M == 0:
            batch_matches.append(matches)
            batch_target_objectness.append(target_objectness)
            batch_target_labels.append(target_labels)
            batch_target_offsets.append(target_offsets)
            continue

        ious = box_ops.box_iou(anchors,gt_boxes) # N,M
        # best_values: N,
        # best_matches: N,
        best_values,best_matches = ious.max(dim=1)
        # best_anchor_match_ids: M,
        _, best_anchor_match_ids = ious.max(dim=0)

        matches[best_values >= pos_iou_tresh] = 1
        matches[best_values < neg_iou_tresh] = -1

        if add_best_matches: matches[best_anchor_match_ids] = 1

        for i in range(M):
            box = gt_boxes[i]
            label = gt_labels[i]
            mask = best_matches == i
            target_offsets[mask] = box
            target_labels[mask] = label

        pos_mask = matches == 1
        # set fg label for objectness
        target_objectness[ pos_mask ] = 1
        target_labels[~pos_mask] = 0

        # convert boxes to offsets
        target_offsets[pos_mask] = boxes2offsets(target_offsets[pos_mask], anchors[pos_mask])

        batch_matches.append(matches)
        batch_target_objectness.append(target_objectness)
        batch_target_labels.append(target_labels)
        batch_target_offsets.append(target_offsets)

    batch_matches = torch.cat(batch_matches, dim=0)
    batch_target_labels = torch.cat(batch_target_labels, dim=0)
    batch_target_objectness = torch.cat(batch_target_objectness, dim=0)
    batch_target_offsets = torch.cat(batch_target_offsets, dim=0)

    return batch_matches, batch_target_objectness, batch_target_labels, batch_target_offsets

def sample_fg_bg(matches, total_samples:int, positive_ratio:float):
    positives, = torch.where(matches == 1)
    negatives, = torch.where(matches == -1)

    num_pos = positives.size(0)
    num_neg = negatives.size(0)

    num_pos = min(int(total_samples * positive_ratio), num_pos)
    num_neg = min(total_samples-num_pos, num_neg)

    selected_pos = torch.randperm(positives.size(0), device=positives.device)[:num_pos]
    selected_neg = torch.randperm(negatives.size(0), device=negatives.device)[:num_neg]

    return positives[selected_pos],negatives[selected_neg]