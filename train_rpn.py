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

def custom_collate_fn(batch):
    batch,targets = zip(*batch)
    return torch.cat(batch,dim=0),targets

def generate_dl(ds, batch_size:int=1, collate_fn=custom_collate_fn,
        num_workers:int=1, pin_memory:bool=True, **kwargs):

    return DataLoader(ds, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=True, **kwargs)

class TrainTransforms():
    def __init__(self, small_dim_size:int=600):
        self.small_dim_size = small_dim_size

    def __call__(self, img, targets:Dict={}):
        # h,w,c => 1,c,h,w
        data = (torch.from_numpy(img).float() / 255).permute(2,0,1).unsqueeze(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = 600 / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'classes' in targets:
            targets['classes'] = torch.from_numpy(targets['classes'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets


def main(dataset_name:str):
    train_transforms = TrainTransforms()

    debug = False
    ds = ds_factory(dataset_name, transforms=train_transforms, download=True)
    dl = generate_dl(ds)

    backbone = models.vgg16(pretrained=True).features[:-1]

    rpn = RPN(backbone, features=512, n=3, effective_stride=16)
    rpn.debug = debug
    rpn.to('cuda')

    running_loss = 0
    rep_count = 10
    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    for iter_count,(batch,targets) in enumerate(dl):
        if debug:
            imgs = [(b*255).long().permute(1,2,0).numpy().astype(np.uint8) for b in batch]
        else:
            imgs = [None] * batch.size(0)

        optimizer.zero_grad()
        loss = rpn.training_step(batch.cuda(), targets, imgs=imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (iter_count+1) % rep_count == 0:
            print("loss: ",running_loss/rep_count)
            running_loss = 0

if __name__ == "__main__":
    main("VOC_train")
