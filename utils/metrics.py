from typing import List,Tuple
import torch
from .box import jaccard_vectorized
from tqdm import tqdm

def calculate_mAP(predictions:List, ground_truths:List,
        num_classes:int, iou_threshold:float=.5, mean:bool=True) -> torch.Tensor:
    """Calculates mean average precision

    Arguments:
        predictions {List} -- [Ni,6 dimensional as xmin,ymin,xmax,ymax,conf,cls_id]
        ground_truths {List} -- [Ni,5 dimensional as xmin,ymin,xmax,ymax,cls_id]
        num_classes {int} -- total number of classes
        iou_threshold {float} -- iou threshold to decide true positive (default: {.5})
        mean {bool} -- if true returns mean of average precisions else returns without mean (default: {True})

    Returns:
        torch.Tensor -- mean average precision
    """
    assert len(predictions) == len(ground_truths),"prediction and ground truths must be equal in lenght"

    mAP = torch.zeros(num_classes, dtype=torch.float32)
    for i in tqdm(range(num_classes), total=num_classes, desc="calculating mAP"):
        cls_predictions = []
        cls_ground_truths = []
        for pred, gt in zip(predictions,ground_truths):
            cls_predictions.append(pred[pred[:, -1] == i][:, :5])
            cls_ground_truths.append(gt[gt[:, -1] == i][:, :4])

        mAP[i] = calculate_AP(cls_predictions,cls_ground_truths,iou_threshold=iou_threshold)

    return mAP.mean() if mean else mAP

def calculate_AP(predictions:List, ground_truths:List, iou_threshold:float=.5) -> torch.Tensor:
    """Calculates average precision

    Arguments:
        predictions {List} -- [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
        ground_truths {List} -- [Ni,4 dimensional as xmin,ymin,xmax,ymax]

    Keyword Arguments:
        iou_threshold {float} -- iou threshold to decide true positive (default: {.5})

    Returns:
        torch.Tensor -- [description]
    """
    sorted_table,M = generate_prediction_table(predictions,ground_truths) # N,3 as iou,best,confidence with sorted by confidence
    N = sorted_table.size(0)

    if N == 0:
        return torch.tensor([0], dtype=torch.float32)

    accumulated_tp = torch.zeros(sorted_table.size(0), dtype=torch.float32)
    accumulated_fp = torch.zeros(sorted_table.size(0), dtype=torch.float32)

    sorted_table[sorted_table[:, 0] < iou_threshold, 1] = 0.
    tp = 0
    fp = 0
    for i,row in enumerate(sorted_table):
        # row : 3 as iou,tp,confidence
        if row[1] == 1.:
            tp += 1
        else:
            fp += 1

        accumulated_tp[i] = tp
        accumulated_fp[i] = fp

    precision = accumulated_tp / torch.arange(1,N+1, dtype=torch.float32)
    recall = accumulated_tp / (M + 1e-16)

    unique_recalls = recall.unique_consecutive()
    auc = torch.empty(unique_recalls.size(0), dtype=torch.float32)
    last_value = torch.tensor(0, dtype=torch.float32)

    for i,recall_value in enumerate(unique_recalls):
        mask = recall == recall_value # N,
        p_mul = precision[mask].max() # get max p
        auc[i] = p_mul * (recall_value-last_value)
        last_value = recall_value

    return auc.sum()

def generate_prediction_table(predictions:List, ground_truths:List) -> Tuple[torch.Tensor,int]:
    """Generates prediction table

    Arguments:
        predictions {List} -- [ni,5 as xmin,ymin,xmax,ymax,confidence] total of N prediction (n0 + n1 + n2 ...)
        ground_truths {List} -- [mi,4 as xmin,ymin,xmax,ymax] total of M ground truths (m0 + m1 + m2 ...)

    Returns:
        Tuple
            torch.Tensor -- N,3 dimensional matrix as iou,best,confidence
            M -- total gt count
    """

    table = []
    M = 0
    for pred,gt in zip(predictions,ground_truths):
        mi = gt.size(0)
        ni = pred.size(0)
        if mi == 0:
            if ni != 0:
                tb = torch.zeros(ni,3, dtype=torch.float32)
                tb[:, 2] = pred[:, 4]
                table.append(tb)
            continue
        elif ni == 0:
            M += mi
            continue
        M += mi
        ious = jaccard_vectorized(pred[:,:4],gt) # ni,mi vector
        iou_values,iou_indexes = ious.max(dim=1)
        ious = torch.stack([iou_values,iou_indexes.float(), pred[:, 4]]).t() # ni,3
        sort_pick = ious[:,0].argsort(dim=0,descending=True) # ni,3
        ious = ious[sort_pick].contiguous() # ni,3
        tb = ious.clone() # ni,3
        mask = [True for i in range(gt.size(0))] # mi,
        for i,iou in enumerate(ious):
            index = int(iou[1].long())
            if mask[index]:
                tb[i][1] = 1.   # assign best
                mask[index] = False
            else:
                tb[i][1] = 0.   # assign ignore
        table.append(tb) # ni,3

    if len(table) == 0:
        return torch.empty(0,3),M

    # return N,3 tensor as iou_value,best,confidence
    table = torch.cat(table,dim=0)
    select = table[:, 2].argsort(descending=True)

    return table[select].contiguous(),M

def caclulate_means(total_metrics):
    means = {}
    """
        [
            {'loss':0, 'a_loss':1,}
        ]
    """
    for metrics in total_metrics:
        for k,v in metrics.items():
            if k in means:
                means[k].append(v)
            else:
                means[k] = [v]

    for k,v in means.items():
        means[k] = sum(means[k]) / len(means[k])
    return means