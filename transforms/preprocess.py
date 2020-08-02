import torch
import torch.nn.functional as F
from typing import Dict,List
from .randomflip import RandomHorizontalFlip
import torchvision.transforms as T
from .padding import Padding
from .interpolate import Interpolate
from typing import Tuple

class TrainTransforms():
    def __init__(self, img_dims:Tuple=(640,800)):
        self.hflip = RandomHorizontalFlip(0.5)
        self.to_tensor = T.ToTensor()
        self.interpolate = Interpolate(img_dims)
        self.padding = Padding(img_dims)

    def __call__(self, img, targets:Dict):
        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']).float() # to float32
        if 'labels' in targets:
            targets['labels'] = torch.from_numpy(targets['labels']).long() # to int64

        img,targets = self.hflip(img,targets)
        data = self.to_tensor(img)
        data.unsqueeze_(0)

        data,targets = self.interpolate(data, targets=targets)
        data,targets = self.padding(data, targets=targets)

        return data,targets

class TestTransforms():
    def __init__(self, img_dims:Tuple=(640,800)):
        self.to_tensor = T.ToTensor()
        self.interpolate = Interpolate(img_dims)
        self.padding = Padding(img_dims)

    def __call__(self, img, targets:Dict):
        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']).float() # to float32
        if 'labels' in targets:
            targets['labels'] = torch.from_numpy(targets['labels']).long() # to int64

        data = self.to_tensor(img)

        # c,h,w => 1,c,h,w
        data.unsqueeze_(0)
        data,targets = self.interpolate(data, targets=targets)
        data,targets = self.padding(data, targets=targets)

        return data,targets