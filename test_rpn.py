import torchvision.models as models
from rpn import RPN
from datasets import factory as ds_factory
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict
import torch.nn.functional as F
import sys
from utils.metrics import roi_recalls

def custom_collate_fn(batch):
    batch,targets = zip(*batch)
    return torch.cat(batch,dim=0),targets

def generate_dl(ds, batch_size:int=1, collate_fn=custom_collate_fn,
        num_workers:int=1, pin_memory:bool=True, **kwargs):

    return DataLoader(ds, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=True, **kwargs)

class TestTransforms():
    def __init__(self, small_dim_size:int=600):
        self.small_dim_size = small_dim_size
        self.means = torch.tensor([122.7717, 115.9465, 102.9801]) # for RGB

    def __call__(self, img, targets:Dict={}):
        # h,w,c => 1,c,h,w
        data = torch.from_numpy(img).float()
        data -= self.means
        data = data.permute(2,0,1).unsqueeze(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = 600 / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'labels' in targets:
            targets['labels'] = torch.from_numpy(targets['labels'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets

def main(model_path:str):
    test_transforms = TestTransforms()
    st = torch.load(model_path,map_location='cpu')

    ds_test = ds_factory("VOC_val", transforms=test_transforms)
    ds_test,_ = torch.utils.data.random_split(ds_test, [int(len(ds_test)/10), len(ds_test)-int(len(ds_test)/10)])
    dl_test = generate_dl(ds_test,batch_size=1)

    backbone = models.vgg16().features[:-1]

    rpn = RPN(backbone, features=512, n=3, effective_stride=16,
        iou_threshold=0.7, conf_threshold=0.0, keep_n=300)
    rpn.debug = False
    rpn.load_state_dict(st, strict=True)
    rpn.to('cuda')

    rpn.eval()
    total_test_iter = int(len(ds_test) / 1)
    print("running test...")
    predictions = []
    ground_truths = []
    for batch,targets in tqdm(dl_test, total=total_test_iter):
        with torch.no_grad():
            preds = rpn.test_step(batch.cuda(), targets)

        ground_truths.append(targets[0]['boxes'].cpu())
        predictions.append(preds[0].cpu())

    iou_thresholds = torch.arange(0.2, 1.0, 0.05, device=predictions[0].device)
    recalls = roi_recalls(predictions, ground_truths, iou_thresholds=iou_thresholds)
    for recall,iou_threshold in zip(recalls,iou_thresholds):
        print(f"VOC val dataset, RPN recall at iou threshold {iou_threshold:.03f} is: {int(recall*100)}")
    print(f"VOC val dataset, RPN mean recall at iou thresholds {iou_thresholds} is: {int(recalls.mean()*100)}")

if __name__ == "__main__":
    main(sys.argv[1])
