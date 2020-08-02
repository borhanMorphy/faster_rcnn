from typing import Tuple
import torch.nn.functional as F

class Padding():
    def __init__(self, image_dims:Tuple, pad_value:int=0):
        # height,width
        self.image_dims = image_dims
        self.pad_value = pad_value

    def __call__(self, batch, targets=None):
        """
            batch: bs x c x h x w
        """
        h,w = batch.shape[-2:]

        if self.image_dims[0] > h:
            pad_top = int((self.image_dims[0] - h) // 2)
            pad_bottom = self.image_dims[0] - h - pad_top
        else:
            pad_top,pad_bottom = (0,0)

        if self.image_dims[1] > w:
            pad_left = int((self.image_dims[1] - w) // 2)
            pad_right = self.image_dims[1] - w - pad_left
        else:
            pad_left,pad_right = (0,0)

        batch = F.pad(batch, (pad_left,pad_right,pad_top,pad_bottom), value=self.pad_value)

        if targets is None:
            return batch

        if "boxes" in targets:
            targets["boxes"][:, 0] += pad_left   # N,(x1,y1,x2,y2)
            targets["boxes"][:, 2] += pad_left   # N,(x1,y1,x2,y2)

            targets["boxes"][:, 1] += pad_top   # N,(x1,y1,x2,y2)
            targets["boxes"][:, 3] += pad_top   # N,(x1,y1,x2,y2)

        return batch, targets

    def __repr__(self):
        return f"Padding({self.image_dims},{self.pad_value})"