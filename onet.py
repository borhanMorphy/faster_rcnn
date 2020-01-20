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
            threshold:float=0.7,iou_threshold:float=0.5,
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
            kernel_size=(2,2),stride=2,padding=0)

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
        # crop bboxes from image and resize to 48x48x3
        batch = []
        h,w = image.shape[:2]
        data = self.preprocess(image) # h,w,c => c,h,w

        bboxes[bboxes[:,0] < 0 , 0] = 0
        bboxes[bboxes[:,1] < 0 , 1] = 0
        bboxes[bboxes[:,2] > w , 2] = w
        bboxes[bboxes[:,3] > h , 3] = h
        
        for bbox in torch.split(bboxes,1,dim=0):
            x1,y1,x2,y2 = bbox[0].int()
            if y2-y1 <= 0 or x2-x1 <= 0:
                continue
            face = F.interpolate(data[:,:,y1:y2,x1:x2],size=(48,48)) # TODO maybe roi pool?
            batch.append(face)
        batch = torch.cat(batch)
        
        # feed forward
        batch = self.conv1(batch)
        batch = self.prelu1(batch)
        batch = self.max_pool1(batch)
        
        batch = self.conv2(batch)
        batch = self.prelu2(batch)
        batch = self.max_pool2(batch)
        
        batch = self.conv3(batch)
        batch = self.prelu3(batch)
        batch = self.max_pool3(batch)
        
        batch = self.conv4(batch)
        batch = self.prelu4(batch)
        batch = self.flatten(batch)
        
        batch = self.linear5(batch)
        batch = self.prelu5(batch)

        cls,reg = self.linear6a(batch),self.linear6b(batch)#,self.linear6c(batch)
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
        #print(x1,y1,x2,y2)
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("",img)
    print(f"avg inference time for pnet: {(time.time()-start_time)/try_count}")
    cv2.waitKey(0)

if __name__ == '__main__':
    import cv2,sys
    import time
    img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_RGB2BGR)
    print("original: ",img.shape)

    from pnet import Pnet
    from rnet import Rnet
    import cv2
    model_pnet = Pnet()
    model_rnet = Rnet()
    model_onet = Onet(threshold=0.88)

    try_count = 1
    start_time = time.time()
    #for i in range(try_count):
    bboxes = model_pnet(img)
    show(bboxes.numpy().tolist(),img.copy())
    #print("after pnet: ",bboxes.size())
    bboxes = model_rnet(bboxes,img)
    show(bboxes.numpy().tolist(),img.copy())
    #print("after rnet: ",bboxes.size())
    bboxes = model_onet(bboxes,img)
    show(bboxes.numpy().tolist(),img.copy())
    #print("after onet: ",bboxes.size())
