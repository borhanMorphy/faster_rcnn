import torchvision.models as models
from rpn import RPN
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F
import sys
from transforms import InferenceTransforms

def load_data(img_path:str):
    oimg = cv2.imread(img_path)
    img = cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)
    return oimg,img

def main(model_path:str, img_path:str):
    test_transforms = InferenceTransforms()
    img,data = load_data(img_path)
    data = test_transforms(data)

    st = torch.load(model_path,map_location='cpu')

    backbone = models.alexnet().features[:-1]

    rpn = RPN(backbone, features=256, n=3, effective_stride=16,
        iou_threshold=0.7, conf_threshold=0.0, keep_n=300)
    rpn.debug = False
    rpn.load_state_dict(st, strict=True)
    rpn.to('cuda')
    rpn.eval()
    with torch.no_grad():
        preds,regs = rpn(data.cuda())
        dets = rpn.detection_layer._inference_postprocess(preds, regs, torch.tensor([*img.shape[:2]]).cuda())

    for x1,y1,x2,y2 in dets[0][:,:4].cpu().long().numpy():
        img2 = cv2.rectangle(img.copy(), (x1,y1),(x2,y2), (0,0,255), 1)
        cv2.imshow("",img2)
        cv2.waitKey(200)

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
