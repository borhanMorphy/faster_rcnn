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
            threshold:float=0.7,iou_threshold:float=0.5,
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

        # normalizing
        data = (data - 127.5) * 0.0078125

        # converting n,h,w,c => n,c,h,w
        data = data.permute(0,3,1,2)
        
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
        # crop bboxes from image and resize to 24x24x3
        batch = []
        h,w = image.shape[:2]
        data = self.preprocess(image) # h,w,c => n,c,h,w
        
        for x1,y1,x2,y2 in bboxes:
            x1 = int(max(0,x1))
            y1 = int(max(0,y1))
            x2 = int(min(w,x2))
            y2 = int(min(h,y2))
            face = F.interpolate(data[:,:,y1:y2,x1:x2],size=(24,24)) # TODO maybe roi pool?
            batch.append(face)
        batch = torch.cat(batch)
        
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

        # apply threshold filter
        pick, = torch.where(cls[:,1]>=self.threshold)
        
        cls = cls[pick,1]
        reg = reg[pick,:]
        bboxes = bboxes[pick,:]

        # apply bbox regresssion
        bboxes = self._bbox_regression(bboxes,reg)

        # apply nms
        pick = torchvision.ops.nms(bboxes,cls,self.iou_threshold)

        # return bboxes
        return bboxes[pick,:]

def show(bboxes,img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for x1,y1,x2,y2 in bboxes:
        h,w = img.shape[:2]
        x1 = int(max(0,x1))
        y1 = int(max(0,y1))
        x2 = int(min(w,x2))
        y2 = int(min(h,y2))
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("",img)
    cv2.waitKey(0)

if __name__ == '__main__':
    import cv2,sys
    import time
    img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2RGB)
    print("original: ",img.shape)

    from pnet import Pnet
    import cv2
    model_pnet = Pnet()
    model_rnet = Rnet()

    bboxes = model_pnet(img)
    print(bboxes.size())
    show(bboxes.clone(),img.copy())
    bboxes = model_rnet(bboxes,img)
    print(bboxes.size())
    show(bboxes,img.copy())