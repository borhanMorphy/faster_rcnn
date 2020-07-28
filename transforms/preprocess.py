import torch
import torch.nn.functional as F
from typing import Dict,List
from .randomflip import RandomHorizontalFlip
import torchvision.transforms as T

class TrainTransforms():
    def __init__(self, small_dim_size:int=600):
        self.small_dim_size = small_dim_size
        self.hflip = RandomHorizontalFlip(0.5)
        self.to_tensor = T.ToTensor()

    def __call__(self, img, targets:Dict=None):
        if targets is not None:
            if 'boxes' in targets:
                targets['boxes'] = torch.from_numpy(targets['boxes']).float() # to float32
            if 'labels' in targets:
                targets['labels'] = torch.from_numpy(targets['labels']).long() # to int64

            if 'img_dims' in targets:
                targets['img_dims'] = torch.from_numpy(targets['img_dims']).long() # to int64

        img,targets = self.hflip(img,targets)
        data = self.to_tensor(img)

        # c,h,w => 1,c,h,w
        data.unsqueeze_(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = self.small_dim_size / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        if targets is None:
            return data

        if 'boxes' in targets:
            targets['boxes'] = targets['boxes'] * scale_factor

        if 'labels' in targets:
            targets['labels'] = targets['labels']

        if 'img_dims' in targets:
            targets['img_dims'] = (targets['img_dims'] * scale_factor).long()

        return data,targets

class TestTransforms():
    def __init__(self, small_dim_size:int=600):
        self.small_dim_size = small_dim_size
        self.to_tensor = T.ToTensor()

    def __call__(self, img, targets:Dict=None):
        if targets is not None:
            if 'boxes' in targets:
                targets['boxes'] = torch.from_numpy(targets['boxes']).float() # to float32
            if 'labels' in targets:
                targets['labels'] = torch.from_numpy(targets['labels']).long() # to int64
            if 'img_dims' in targets:
                targets['img_dims'] = torch.from_numpy(targets['img_dims']).long()

        data = self.to_tensor(img)

        # c,h,w => 1,c,h,w
        data.unsqueeze_(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = self.small_dim_size / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        if targets is None:
            return data

        if 'boxes' in targets:
            targets['boxes'] = targets['boxes'] * scale_factor

        if 'labels' in targets:
            targets['labels'] = targets['labels']

        if 'img_dims' in targets:
            targets['img_dims'] = (targets['img_dims'] * scale_factor).long()

        return data,targets