import torch
import torch.nn.functional as F
from typing import Dict,List

class TrainTransforms():
    def __init__(self, small_dim_size:int=600, mean:float=.0, std:float=255.0):
        self.small_dim_size = small_dim_size
        self.mean = mean
        self.std = std

    def __call__(self, img, targets:Dict=None):
        # h,w,c => 1,c,h,w
        data = (torch.from_numpy(img).float() - self.mean) / self.std
        data = data.permute(2,0,1).unsqueeze(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = self.small_dim_size / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        if targets is None:
            return data

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'labels' in targets:
            targets['labels'] = torch.from_numpy(targets['labels'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets

class TestTransforms():
    def __init__(self, small_dim_size:int=600, mean:float=.0, std:float=255.0):
        self.small_dim_size = small_dim_size
        self.mean = mean
        self.std = std

    def __call__(self, img, targets:Dict=None):
        # h,w,c => 1,c,h,w
        data = (torch.from_numpy(img).float() - self.mean) / self.std
        data = data.permute(2,0,1).unsqueeze(0)
        h = data.size(2)
        w = data.size(3)
        scale_factor = 600 / min(h,w)
        data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        if targets is None:
            return data

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'labels' in targets:
            targets['labels'] = torch.from_numpy(targets['labels'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets

class InferenceTransforms():
    def __init__(self, mean:float=.0, std:float=255.0):
        self.mean = mean
        self.std = std

    def __call__(self, img, targets:Dict=None):
        # h,w,c => 1,c,h,w
        data = (torch.from_numpy(img).float() - self.mean) / self.std
        data = data.permute(2,0,1).unsqueeze(0)
        if targets is None:
            return data

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes'])

        if 'labels' in targets:
            targets['labels'] = torch.from_numpy(targets['labels'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims'])).long()

        return data,targets