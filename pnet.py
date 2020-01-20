import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision

class Pnet(nn.Module):
    def __init__(self,threshold:float=0.6,
            iou_threshold:float=0.5,model_path:str="pnet.pth",
            gpu:int=-1,training:bool=False):
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

        self.softmax = nn.Softmax2d()

        state_dict = torch.load(model_path,map_location=self._device)
        self.load_state_dict(state_dict)
        self.requires_grad_(training)
        self.to(self._device)
        self.training = training
        if training:
            self.train()
        else:
            self.eval()
        
    def _gen_prediction_results(self,ground_truths:torch.Tensor,
            predictions:torch.Tensor,scores:torch.Tensor):
        """label each prediction as true positive or false positive using ground truths
        
        Arguments:
            ground_truths {torch.Tensor} -- N,4 boxes with order of x1,y1,x2,y2
            predictions {torch.Tensor} -- N,4 boxes with order of x1,y1,x2,y2
            scores {torch.Tensor} -- N scores of boxes
        """
        pass

    def _rerec(self,bbox:torch.Tensor):
        # convert bbox to square
        height = bbox[:, 3] - bbox[:, 1]
        width = bbox[:, 2] - bbox[:, 0]
        dim = torch.stack([width,height])
        max_side_length,max_side_indexes = dim.max(dim=0)
        bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
        bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + max_side_length.repeat(2, 1).t()
        return bbox

    def forward(self,img:np.ndarray,**kwargs):
        # calculate scales
        scales = self.scale_pyramid(img.shape[:2])
        
        boxes,scores,regressions = [],[],[]
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
            cls = self.softmax(cls)

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
            
            bboxes = windows#self._xywh2xyxy(windows)
            
            if not self.training:
                # apply nms
                pick = torchvision.ops.nms(bboxes,cls,self.iou_threshold)
                bboxes = bboxes[pick,:]
                cls = cls[pick]
                reg = reg[pick]

            boxes.append(bboxes)
            scores.append(cls)
            regressions.append(reg)
        
        boxes = torch.cat(boxes,dim=0)
        scores = torch.cat(scores,dim=0)
        regressions = torch.cat(regressions,dim=0)


        if not self.training:
            # apply nms
            pick = torchvision.ops.nms(boxes,scores,self.iou_threshold+0.2)
            boxes = boxes[pick,:]

            # apply bbox regression
            boxes = self._bbox_regression(boxes,regressions)

            # bbox to rect
            boxes = self._rerec(boxes)

            # drop
            boxes = self._drop_unvalid_boxes(boxes)
        else:
            # generate prediction result using ground truths
            self._gen_prediction_results()

        return boxes

    def _bbox_regression(self,windows:torch.Tensor,reg:torch.Tensor):
        """bounding box regression
        
        Arguments:
            windows {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
            reg {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
        
        Returns:
            torch.Tensor -- calibrated bounding boxes with order of x1,y1,x2,y2
        """
        windows = windows.to(self._device)
        
        w = windows[:,2]-windows[:,0]
        h = windows[:,3]-windows[:,1]

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
        
        #offset_x = ((w-self.offset*2)%self.stride)/2 + self.offset
        #offset_y = ((h-self.offset*2)%self.stride)/2 + self.offset
        #
        #cx = x * self.stride+offset_x
        #cy = y * self.stride+offset_y
        dims = torch.stack([y,x],dim=0).t()
        qq1 = (dims * self.stride + 1)/scale
        qq2 = (dims * self.stride + 12)/scale
        
        n = y.size()[0]
        if n == 0:
            return

        #windows = torch.cat([self.window.view(-1,2) for _ in range(n)],dim=0)

        #return torch.cat([cx.view(-1,1),cy.view(-1,1),windows],dim=1)/scale
        return torch.cat([qq1,qq2],dim=1)

    def preprocess(self, data, scale:float=1.0):
        
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

        # scale
        data = F.interpolate(data,size=(int(h*scale),int(w*scale)))
        
        return data

    def scale_pyramid(self,dims,min_face_size:int=20,
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
    
    def _drop_unvalid_boxes(self,boxes:torch.Tensor):
        bb = []
        for i,bbox in enumerate(boxes):
            x1,y1,x2,y2 = bbox.int()
            if i == 28:
                print(x1,y1,x2,y2)
            if x2-x1 > 0 and y2-y1 > 0:
                bb.append(bbox)
                print("girdi")
            else:
                print(x1,x2,y1,y2)
        return torch.stack(bb,dim=0)

if __name__ == '__main__':
    import cv2,sys
    import time
    img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2RGB)
    
    model = Pnet(gpu=-1,training=False)
    
    bboxes = model(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for x1,y1,x2,y2 in bboxes.numpy().tolist():
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        temp_img = cv2.rectangle(img.copy(),(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("",temp_img)
        cv2.waitKey(0)


    