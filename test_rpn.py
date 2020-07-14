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
from utils.metrics import calculate_AP

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

    def __call__(self, img, targets:Dict={}):
        # h,w,c => 1,c,h,w
        data = (torch.from_numpy(img).float() / 255).permute(2,0,1).unsqueeze(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = 600 / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'classes' in targets:
            targets['classes'] = torch.from_numpy(targets['classes'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets


def main(model_path:str):
    test_transforms = TestTransforms()

    ds_test = ds_factory("VOC_val", transforms=test_transforms)
    ds_test,_ = torch.utils.data.random_split(ds_test, [int(len(ds_test)/10), len(ds_test)-int(len(ds_test)/10)])
    dl_test = generate_dl(ds_test,batch_size=1)

    backbone = models.vgg16().features[:-1]

    rpn = RPN(backbone, features=512, n=3, effective_stride=16,
        iou_threshold=0.7, conf_threshold=0.5, keep_n=300)
    rpn.debug = False
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
    ap = calculate_AP(predictions,ground_truths, iou_threshold=0.5)
    print(f"test AP score: {ap}")

if __name__ == "__main__":
    main(sys.argv[1])
