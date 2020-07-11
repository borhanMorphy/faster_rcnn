import torch
from cv2 import cv2


def load_data(img_path:str):
    oimg = cv2.imread(img_path)
    data = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(data).float() / 255).permute(2,0,1).unsqueeze(0),oimg