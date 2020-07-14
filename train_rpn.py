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
from utils.metrics import caclulate_means

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
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'classes' in targets:
            targets['classes'] = torch.from_numpy(targets['classes'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets

def reduce_dataset(ds,ratio=0.1):
    size = int(len(ds)*ratio)
    rest = len(ds)-size
    return torch.utils.data.random_split(ds, [size,rest])[0]

def main():
    train_transforms = TrainTransforms()

    debug = False # TODO add debug
    batch_size = 1
    epochs = 1

    # !defined in the paper
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    total_iter_size = 60000

    ds_train = ds_factory("VOC_train", transforms=train_transforms)
    dl_train = generate_dl(ds_train,batch_size=batch_size)

    ds_val = ds_factory("VOC_val", transforms=train_transforms)
    ds_val = reduce_dataset(ds_val,ratio=0.1)
    dl_val = generate_dl(ds_val,batch_size=batch_size)

    backbone = models.vgg16(pretrained=True).features[:-1]

    rpn = RPN(backbone, features=512, n=3, effective_stride=16)
    rpn.debug = debug
    rpn.to('cuda')

    verbose = 100
    optimizer = torch.optim.SGD(rpn.parameters(), lr=learning_rate,
        momentum=momentum, weight_decay=weight_decay)

    max_iter_count = int(len(ds_train)/batch_size)
    # ! set because of the paper
    epochs = int(total_iter_size / max_iter_count)

    for epoch in range(epochs):
        running_metrics = []
        print(f"running epoch [{epoch+1}/{epochs}]")
        rpn.train()
        for iter_count,(batch,targets) in enumerate(dl_train):

            optimizer.zero_grad()
            metrics = rpn.training_step(batch.cuda(), targets)
            metrics['loss'].backward()
            optimizer.step()

            metrics['loss'] = metrics['loss'].item()

            running_metrics.append(metrics)
            if (iter_count+1) % verbose == 0:
                means = caclulate_means(running_metrics)
                running_metrics = []
                log = []
                for k,v in means.items():
                    log.append(f"{k}: {v:.04f}")
                log = "\t".join(log)
                log += f"\titer[{iter_count}/{max_iter_count}]"
                print(log)

        torch.save(rpn.state_dict(), f"./rpn_epoch_{epoch+1}.pth")

        # start validation
        running_metrics = []
        total_val_iter = int(len(ds_val) / batch_size)
        rpn.eval()
        print("running validation...")
        for batch,targets in tqdm(dl_val, total=total_val_iter):
            with torch.no_grad():
                metrics = rpn.training_step(batch.cuda(), targets)
            metrics['loss'] = metrics['loss'].item()
            running_metrics.append(metrics)

        means = caclulate_means(running_metrics)
        log = []
        for k,v in means.items():
            log.append(f"{k}: {v:.04f}")
        log = "\t".join(log)
        print(f"validation results for epoch {epoch+1} is: {log}")

if __name__ == "__main__":
    main()
