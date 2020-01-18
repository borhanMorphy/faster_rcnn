import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Onet(nn.Module):
    def __init__(self,
            threshold:float=0.6,iou_threshold:float=0.5,
            model_path:str="onet.pth",gpu:int=-1):
        super(Onet,self).__init__()
        self._device = torch.device("cpu") if gpu == -1 else torch.device(f"cuda:{gpu}") 
        self.threshold = torch.tensor(threshold).to(self._device)
        self.iou_threshold = torch.tensor(iou_threshold).to(self._device)

        # Layers 
        ############################################
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,
            kernel_size=(3,3),stride=1,padding=0)

        self.prelu1 = nn.PReLU(num_parameters=32)

        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(3,3),stride=2,padding=1)
        ############################################

        ############################################
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,
            kernel_size=(3,3),stride=1,padding=0)
        
        self.prelu2 = nn.PReLU(num_parameters=64)

        self.max_pool2 = nn.MaxPool2d(
            kernel_size=(3,3),stride=2,padding=0)
        ############################################

        ############################################
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,
            kernel_size=(3,3),stride=1,padding=0)
        
        self.prelu3 = nn.PReLU(num_parameters=64)

        self.max_pool3 = nn.MaxPool2d(
            kernel_size=(2,2),stride=2,padding=1)

        ############################################

        ############################################
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,
            kernel_size=(2,2),stride=1,padding=0)
        
        self.prelu4 = nn.PReLU(num_parameters=128)

        self.flatten = Flatten() # 3x3x128 => 1152
        ############################################

        ############################################
        self.linear5 = nn.Linear(1152,256)

        self.prelu5 = nn.PReLU(num_parameters=256)
        ############################################


        ############################################
        self.linear6a = nn.Linear(256,2)

        self.softmax = nn.Softmax(dim=1)
        ############################################


        ############################################
        self.linear6b = nn.Linear(256,4)
        ############################################
        
        ############################################
        self.linear6c = nn.Linear(256,10)
        ############################################

        state_dict = torch.load(model_path,map_location=self._device)
        self.load_state_dict(state_dict)
        self.requires_grad_(False)
        self.to(self._device)
        self.eval()

    def preprocess(self, data):
        data = data.copy()
        if len(data.shape) == 3:
            data = np.expand_dims(data,axis=0)
        
        _,h,w,_ = data.shape

        # converting numpy => tensor
        data = torch.from_numpy(data.astype(np.float32)).to(self._device)

        # converting n,h,w,c => n,c,h,w
        data = data.permute(0,3,1,2)

        # normalizing
        data = (data - 127.5) * 0.0078125
        
        return data

    def _bbox_regression(self,windows:torch.Tensor,reg:torch.Tensor):
        """bounding box regression
        
        Arguments:
            windows {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
            reg {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
        
        Returns:
            torch.Tensor -- calibrated bounding boxes with order of x1,y1,x2,y2
        """
        windows = windows.to(self._device)
        w = windows[:,2] - windows[:,0]
        h = windows[:,3] - windows[:,1]

        x1 = windows[:,0] + reg[:,0]*w
        y1 = windows[:,1] + reg[:,1]*h
        x2 = windows[:,2] + reg[:,2]*w
        y2 = windows[:,3] + reg[:,3]*h

        return torch.stack([x1,y1,x2,y2]).t()

    def forward(self,bboxes:torch.Tensor,image:np.ndarray):
        """
        crop bboxes from image
        
        Arguments:
            bboxes {torch.Tensor} -- x1,y1,x2,y2
            image {np.ndarray} -- h,w,c image
        
        Returns:
            [type] -- [description]
        """
        pass

if __name__ == '__main__':
    Onet()