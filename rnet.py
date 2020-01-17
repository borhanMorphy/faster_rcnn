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

        self.flatten = Flatten() # 3x3x64 => 576
        ############################################

        ############################################
        self.linear4 = nn.Linear(576,128)

        self.prelu4 = nn.PReLU(num_parameters=128)
        ############################################


        ############################################
        self.linear5a = nn.Linear(128,2)

        self.softmax = nn.Softmax(dim=1)
        ############################################


        ############################################
        self.linear5b = nn.Linear(128,4)
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

    def forward(self,bboxes:torch.Tensor,image:np.ndarray):
        """
        crop bboxes from image
        
        Arguments:
            bboxes {torch.Tensor} -- x1,y1,x2,y2
            image {np.ndarray} -- h,w,c image
        
        Returns:
            [type] -- [description]
        """
        # crop bboxes from image and resize to 24x24x3
        batch = []
        h,w = image.shape[:2]
        data = self.preprocess(image) # h,w,c => n,c,h,w

        bboxes[bboxes[:,0] < 0 , 0] = 0
        bboxes[bboxes[:,1] < 0 , 1] = 0
        bboxes[bboxes[:,2] > w , 2] = w
        bboxes[bboxes[:,3] > h , 3] = h
        
        for bbox in torch.split(bboxes,1,dim=0):
            x1,y1,x2,y2 = bbox[0].int()
            face = F.interpolate(data[:,:,y1:y2,x1:x2],size=(24,24)) # TODO maybe roi pool?
            batch.append(face)
        batch = torch.cat(batch)
        print(batch.size())
        batch_size = batch.size(0)

        # feed forward
        batch = self.conv1(batch)
        batch = self.prelu1(batch)
        batch = self.max_pool1(batch)

        batch = self.conv2(batch)
        batch = self.prelu2(batch)
        batch = self.max_pool2(batch)

        batch = self.conv3(batch)
        batch = self.prelu3(batch)
        batch = self.flatten(batch)

        batch = self.linear4(batch)
        batch = self.prelu4(batch)

        cls,reg = self.linear5a(batch),self.linear5b(batch)
        cls = self.softmax(cls)

        print("cls: ",cls.size())
        print("reg: ",reg.size())
        # apply threshold filter

        # apply bbox regresssion

        # apply nms

        # return bboxes

if __name__ == '__main__':
    import cv2,sys
    import time
    img = cv2.imread(sys.argv[1])
    print("original: ",img.shape)

    from pnet import Pnet
    import cv2
    model_pnet = Pnet()
    model_rnet = Rnet()

    bboxes = model_pnet(img)
    model_rnet(bboxes,img)