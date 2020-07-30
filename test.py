import torch
from typing import List

def generate_base_anchors(sizes:List[int], aspect_ratios:List[float]):
    sizes = torch.tensor(sizes, dtype=torch.float32)
    aspect_ratios = torch.tensor(aspect_ratios, dtype=torch.float32)
    
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    ws = (w_ratios.unsqueeze(0).t() * sizes.unsqueeze(0)).view(-1)
    hs = (h_ratios.unsqueeze(0).t() * sizes.unsqueeze(0)).view(-1)

    return (torch.stack([-ws, -hs, ws, hs], dim=1) / 2).round()


def calc_shifts(img_dim=(960,640), fmap_dim=(30,20), dtype=torch.float32, device='cpu'):
    base_anchors = torch.cat([generate_base_anchors([size],[0.5,1,2]) for size in [128,256,512]])
    h,w = img_dim
    fh,fw = fmap_dim
    stride = torch.tensor([h/fh, w/fw], dtype=torch.int64, device=device)

    shift_x = torch.arange(0, fw, dtype=dtype, device=device) * int(w/fw)
    shift_y = torch.arange(0, fh, dtype=dtype, device=device) * int(h/fh)
    shift_y,shift_x = torch.meshgrid(shift_y,shift_x)

    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

    anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)

if __name__ == "__main__":
    calc_shifts()