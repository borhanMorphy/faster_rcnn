import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision

class Pnet(nn.Module):
    def __init__(self,
            threshold:float=0.6,iou_threshold:float=0.5,
            model_path:str="pnet.pth",gpu:int=-1):
        super(Pnet,self).__init__()
        self._device = torch.device("cpu") if gpu == -1 else torch.device(f"cuda:{gpu}") 
        self.threshold = torch.tensor(threshold).to(self._device)
        self.iou_threshold = torch.tensor(iou_threshold).to(self._device)
        
        self.stride = 2
        self.offset = 6
        self.window = torch.tensor([12,12],dtype=torch.float32)

        # Layers 
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=10,
            kernel_size=(3,3),stride=1,padding=0)

        self.prelu1 = nn.PReLU(num_parameters=10)

        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(2,2),stride=2,padding=0)

        self.conv2 = nn.Conv2d(in_channels=10,out_channels=16,
            kernel_size=(3,3),stride=1,padding=0)
        
        self.prelu2 = nn.PReLU(num_parameters=16)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,
            kernel_size=(3,3),stride=1,padding=0)

        self.prelu3 = nn.PReLU(num_parameters=32)

        self.conv4a = nn.Conv2d(in_channels=32,out_channels=2,
            kernel_size=(1,1),stride=1,padding=0)

        self.conv4b = nn.Conv2d(in_channels=32,out_channels=4,
            kernel_size=(1,1),stride=1,padding=0)

        self.softmax1 = nn.Softmax2d()

        state_dict = torch.load(model_path,map_location=self._device)
        self.load_state_dict(state_dict)
        self.requires_grad_(False)
        self.to(self._device)
        self.eval()

    def forward(self,img:np.ndarray):
        # calculate scales
        scales = self.scale_pyramid(img.shape[:2])
        
        boxes,scores = [],[]
        for scale in scales:
            # preprocess
            x = self.preprocess(img,scale)
            # extract w and h
            h,w = x.size()[-2:]

            # forward with conv net
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.max_pool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            cls,reg = self.conv4a(x),self.conv4b(x)
            cls = self.softmax1(cls)

            # transform
            # regression batch_size,4,h,w => batch_size,h,w,4
            reg = reg.permute(0,2,3,1)

            # classification batch_size,2,h,w => batch_size,h,w,2
            cls = cls.permute(0,2,3,1)

            # filter with threshold
            y,x = torch.where(cls[0,:,:,1]>=self.threshold)
            
            cls = cls[0,y,x,1]
            reg = reg[0,y,x,:]

            # generate windows
            windows = self._gen_windows((y,x),(h,w),scale)
            if isinstance(windows,type(None)):
                continue

            # apply bbox regression
            bboxes = self._bbox_regression(windows,reg)

            # apply nms
            pick = torchvision.ops.nms(bboxes,cls,self.iou_threshold)
            
            boxes.append(bboxes[pick,:])
            scores.append(cls[pick])
        
        # cat
        boxes = torch.cat(boxes,dim=0)
        scores = torch.cat(scores,dim=0)
        # apply nms
        pick = torchvision.ops.nms(boxes,scores,self.iou_threshold+0.2)

        return boxes[pick,:]

    def _bbox_regression(self,windows:torch.Tensor,reg:torch.Tensor):
        """bounding box regression
        
        Arguments:
            windows {torch.Tensor} -- [N,4] with order of cx,cy,w,h
            reg {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
        
        Returns:
            torch.Tensor -- calibrated bounding boxes with order of x1,y1,x2,y2
        """
        windows = windows.to(self._device)
        w,h = windows[:,2],windows[:,3]
        # transform cx cy w h => x1 y1 x2 y2
        windows = self._xywh2xyxy(windows)

        x1 = windows[:,0] + reg[:,0]*w
        y1 = windows[:,1] + reg[:,1]*h
        x2 = windows[:,2] + reg[:,2]*w
        y2 = windows[:,3] + reg[:,3]*h

        return torch.stack([x1,y1,x2,y2]).t()

    def _xywh2xyxy(self,boxes):
        """Transforms center x, center y, width ,height to x1,y1,x2,y2
        """
        x1 = boxes[:,0]-boxes[:,2]/2
        y1 = boxes[:,1]-boxes[:,3]/2
        x2 = boxes[:,0]+boxes[:,2]/2
        y2 = boxes[:,1]+boxes[:,3]/2
        return torch.stack([x1,y1,x2,y2]).t()


    def _gen_windows(self,selected_indexes,dims,scale):
        bboxes = []
        h,w = dims
        y,x = selected_indexes
        y = y.cpu().float() if y.is_cuda else y.float()
        x = x.cpu().float() if x.is_cuda else x.float()
        
        offset_x = ((w-self.offset*2)%self.stride)/2 + self.offset
        offset_y = ((h-self.offset*2)%self.stride)/2 + self.offset
        
        cx = x * self.stride+offset_x
        cy = y * self.stride+offset_y
        
        n = y.size()[0]
        if n == 0:
            return

        windows = torch.cat([self.window.view(-1,2) for _ in range(n)],dim=0)
        
        windows /= scale
        
        cx /= scale
        cy /= scale

        return torch.cat([cx.view(-1,1),cy.view(-1,1),windows],dim=1)

    def preprocess(self, data, scale:float=1.0):
        
        data = data.copy()
        if len(data.shape) == 3:
            data = np.expand_dims(data,axis=0)
        
        _,h,w,_ = data.shape
        # converting numpy => tensor
        data = torch.from_numpy(data.astype(np.float32)).to(self._device)

        # converting n,h,w,c => n,c,w,h
        data = data.permute(0,3,2,1)

        # scale
        data = F.interpolate(data,size=(int(w*scale),int(h*scale)))

        # normalizing
        data = (data - 127.5) * 0.0078125
        
        return data

    def scale_pyramid(self,dims,
            min_face_size:int=20,
            scale_factor:float=0.709):
        # TODO handle batch
        h,w = dims
        m = 12 / min_face_size
        min_layer = np.amin([h, w]) * m
        
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(scale_factor, factor_count)]
            min_layer = min_layer * scale_factor
            factor_count += 1

        return scales

if __name__ == '__main__':
    import cv2,sys
    import time
    img = cv2.imread(sys.argv[1])
    
    model = Pnet(gpu=-1)
    try_count = 1
    start_time = time.time()
    for i in range(try_count):
        bboxes = model(img)
        """
        for x1,y1,x2,y2 in bboxes.cpu().numpy().tolist():
            h,w = img.shape[:2]
            x1 = int(max(0,x1))
            y1 = int(max(0,y1))
            x2 = int(min(w,x2))
            y2 = int(min(h,y2))
            print(x1,y1,x2,y2)
            im = img.copy()
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.imshow("",im)
            cv2.waitKey(0)
        """
    print(f"avg inference time for pnet: {(time.time()-start_time)/try_count}")


    