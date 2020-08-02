from typing import Tuple
import torch.nn.functional as F

class Interpolate():
    def __init__(self, image_dims:Tuple):
        # height,width
        self.image_dims = image_dims

    def __call__(self, batch, targets=None):
        """
            batch: bs x c x h x w
        """
        h,w = batch.shape[-2:]

        # interpolate
        scale_factor = min(self.image_dims[0] / h, self.image_dims[1] / w)

        batch = F.interpolate(batch,
            scale_factor=scale_factor,
            mode='bilinear', recompute_scale_factor=False, align_corners=False)

        if targets is None:
            return batch

        if "boxes" in targets:
            targets["boxes"] *= scale_factor  # N,(x1,y1,x2,y2)

        return batch,targets

    def __repr__(self):
        return f"Interpolate({self.image_dims})"

if __name__ == "__main__":
    t = Interpolate((600,800))

    import torch
    i = torch.rand(1,3,600,850)

    print(t(i).shape)