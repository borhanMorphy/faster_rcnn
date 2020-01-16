import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Rnet(nn.Module):
    def __init__(self,
            threshold:float=0.6,iou_threshold:float=0.5,
            model_path:str="rnet.pth",gpu:int=-1):
        super(Rnet,self).__init__()
        self._device = torch.device("cpu") if gpu == -1 else torch.device(f"cuda:{gpu}") 
        self.threshold = torch.tensor(threshold).to(self._device)
        self.iou_threshold = torch.tensor(iou_threshold).to(self._device)

        # Layers 
        ############################################
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=28,
            kernel_size=(3,3),stride=1,padding=0)

        self.prelu1 = nn.PReLU(num_parameters=28)

        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(3,3),stride=2,padding=1)
        ############################################

        ############################################
        self.conv2 = nn.Conv2d(in_channels=28,out_channels=48,
            kernel_size=(3,3),stride=1,padding=0)
        
        self.prelu2 = nn.PReLU(num_parameters=48)

        self.max_pool2 = nn.MaxPool2d(
            kernel_size=(3,3),stride=2,padding=0)
        ############################################

        ############################################
        self.conv3 = nn.Conv2d(in_channels=48,out_channels=64,
            kernel_size=(2,2),stride=1,padding=0)
        
        self.prelu3 = nn.PReLU(num_parameters=64)

        self.flatten = Flatten()
        ############################################

        ############################################
        self.linear4 = nn.Linear(64,128)

        self.prelu4 = nn.PReLU(num_parameters=128)
        ############################################


        ############################################
        self.linear5a = nn.Linear(128,2)

        self.softmax = nn.Softmax2d()
        ############################################


        ############################################
        self.linear5b = nn.Linear(128,4)
        ############################################

        state_dict = torch.load(model_path,map_location=self._device)
        self.load_state_dict(state_dict)
        self.requires_grad_(False)
        self.to(self._device)
        self.eval()

    def forward(self,candidates:torch.Tensor,image:np.ndarray):
        pass
        # crop candidates from image and resize to 24x24x3

        # feed forward

        # apply threshold filter

        # apply bbox regresssion

        # apply nms

        # return bboxes