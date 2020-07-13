import torchvision.models as models
from rpn import RPN
from datasets import factory as ds_factory
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict

def custom_collate_fn(batch):
    batch,targets = zip(*batch)
    return torch.stack(batch),targets

def generate_dl(ds, batch_size:int=1, collate_fn=custom_collate_fn,
        num_workers:int=1, pin_memory:bool=True, **kwargs):

    return DataLoader(ds, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=True, **kwargs)

class TrainTransforms():
    def __init__(self):
        pass

    def __call__(self, img, targets:Dict={}):
        data = (torch.from_numpy(img).float() / 255).permute(2,0,1)
        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes'])

        if 'classes' in targets:
            targets['classes'] = torch.from_numpy(targets['classes'])

        return data,targets

def main(dataset_name:str):
    train_transforms = TrainTransforms()

    debug = True
    ds = ds_factory(dataset_name, transforms=train_transforms)
    dl = generate_dl(ds)

    backbone = models.vgg16(pretrained=True).features[:-1]
    backbone.eval()

    rpn = RPN(features=512, n=3, effective_stride=16)
    rpn.debug = debug
    imgs = None

    for batch,targets in dl:
        if debug:
            imgs = [(b*255).long().permute(1,2,0).numpy().astype(np.uint8) for b in batch]

        with torch.no_grad():
            fmap = backbone(batch)

        loss = rpn.training_step(fmap, targets, imgs=imgs)

if __name__ == "__main__":
    main("VOC_train")
