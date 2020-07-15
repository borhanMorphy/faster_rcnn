import torchvision.models as models
from rpn import RPN
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F
import sys

def load_data(img_path:str):
    oimg = cv2.imread(img_path)
    #img = cv2.cvtColor(oimg,cv2.COLOR_BGR2RGB)
    return oimg,oimg.copy()

class TestTransforms():
    def __init__(self, small_dim_size:int=600):
        self.small_dim_size = small_dim_size
        self.means = torch.tensor([102.9801, 115.9465, 122.7717]) 

    def __call__(self, img, targets:Dict={}):
        # h,w,c => 1,c,h,w
        data = torch.from_numpy(img).float()
        data -= self.means
        data = data.permute(2,0,1).unsqueeze(0)
        h = data.size(2)
        w = data.size(3)
        return data
        #scale_factor = 600 / min(h,w)
        #data = F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        if 'boxes' in targets:
            targets['boxes'] = torch.from_numpy(targets['boxes']) * scale_factor

        if 'classes' in targets:
            targets['classes'] = torch.from_numpy(targets['classes'])

        if 'img_dims' in targets:
            targets['img_dims'] = (torch.from_numpy(targets['img_dims']) * scale_factor).long()

        return data,targets

def main(model_path:str, img_path:str):
    test_transforms = TestTransforms()
    img,data = load_data(img_path)
    data = test_transforms(data)

    st = torch.load(model_path,map_location='cpu')

    backbone = models.vgg16().features[:-1]

    rpn = RPN(backbone, features=512, n=3, effective_stride=16,
        iou_threshold=0.5, conf_threshold=0.0, keep_n=300)
    rpn.debug = False
    rpn.load_state_dict(st, strict=True)
    rpn.to('cuda')
    rpn.eval()
    with torch.no_grad():
        preds,regs = rpn(data.cuda())
        dets = rpn.detection_layer._inference_postprocess(preds,regs,torch.tensor([*img.shape[:2]]).cuda())
    
    sort = dets[0][:,4].argsort(descending=True)

    for x1,y1,x2,y2 in dets[0][sort,:4].cpu().long().numpy():
        img2 = cv2.rectangle(img.copy(), (x1,y1),(x2,y2), (0,0,255), 1)
        cv2.imshow("",img2)
        cv2.waitKey(200)
    
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
