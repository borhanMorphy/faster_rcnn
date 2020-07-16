import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.models as models
from cv2 import cv2
from typing import Tuple
import sys,os

class RCNN(nn.Module):
    """ RCNN with fully connected layers but not SVM """
    def __init__(self, backbone, input_size:Tuple=(224,224), spatial_scale:float=1.0):
        super(RCNN,self).__init__()
        self.roi_pool = ops.RoIPool(output_size=input_size, spatial_scale=spatial_scale)
        self.backbone = backbone

    def forward(self, image, rois):
        batch = self.roi_pool(image,rois)
        print(batch.shape)
        preds = self.backbone(batch)
        return preds

def load_data(img_path):
    img = cv2.imread(img_path)
    batch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batch = (torch.from_numpy(batch).float() / 255.0).permute(2,0,1).unsqueeze(0)
    return img,batch

if __name__ == "__main__":
    backbone = models.alexnet(pretrained=True)
    backbone.classifier = nn.Sequential(
        *backbone.classifier[:-1],
        nn.Linear(4096, 2)
    )

    img,batch = load_data(sys.argv[1])

    model = RCNN(backbone)

    res = model(batch)

    print(res.shape)

    cv2.imshow("",img)
    cv2.waitKey(0)