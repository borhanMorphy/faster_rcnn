import torchvision.transforms.functional as F
import random
from typing import Dict,Union
import torch
import numpy as np

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, targets:Dict[str,Union[torch.Tensor,np.ndarray]]=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            targets : contains boxes(xmin,ymin,xmax,ymax) and labels

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            w,_ = img.size
            if targets is not None and 'boxes' in targets:
                boxes = targets['boxes'].clone()
                targets['boxes'][:, 0] = w - boxes[:, 2]
                targets['boxes'][:, 2] = w - boxes[:, 0]

            return F.hflip(img),targets
        return img,targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)