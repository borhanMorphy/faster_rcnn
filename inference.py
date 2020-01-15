import torch
import torchvision
import numpy as np

def preprocess(x:np.ndarray):
    x = x.copy().astype(np.float32)
    x /= 255.0
    x = np.transpose(x,(2,0,1))
    x = np.expand_dims(x,axis=0)
    return torch.from_numpy(x)

class FeatureNetwork(torch.nn.Module):
    def __init__(self):
        super(FeatureNetwork,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features[:-1]

    def forward(self,x,**kwargs):
        return self.features(x,**kwargs)

class RPN(torch.nn.Module):
    def __init__(self,
            threshold:float=0.7,N:int=50,iou_threshold:float=0.7,
            backbone:str="alexnet",model_path:str="vgg16_rpn.pth"):
        super(RPN,self).__init__()
        anchor_ratios = torch.tensor([[1,1],[1,2],[2,1]],dtype=torch.float32) # x:y
        anchor_scales = [128,256,512]
        if backbone == "alexnet":
            self.stride = 16         # for alexnet
            self.offset = 17.5       # for alexnet
            self.fmap_channel = 256  # for alexnet
        elif backbone == "vgg16":
            self.stride = 16         # for vgg16
            self.offset = 8          # for vgg16
            self.fmap_channel = 512  # for vgg16
        else:
            raise ValueError("Backbone must be defined!")
        
        self.anchors = torch.cat(
            [
                anchor_ratios*anchor_scale 
                for anchor_scale in anchor_scales
            ],
            dim=0
        )
        self.anchor_size = self.anchors.size()[0]
        
        
        self.threshold = threshold#torch.tensor([threshold],dtype=torch.float32)
        self._N = N # TOP scored proposal selection count after NMS operation
        self.iou_threshold = iou_threshold

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.fmap_channel,
            out_channels=self.fmap_channel,
            kernel_size=3,stride=1,padding=1)
        
        self.conv2_cls = torch.nn.Conv2d(
            in_channels=self.fmap_channel,
            out_channels=2*self.anchor_size,
            kernel_size=1,stride=1)

        self.softmax = torch.nn.Softmax2d()
        
        self.conv2_reg = torch.nn.Conv2d(
            in_channels=self.fmap_channel,
            out_channels=4*self.anchor_size,
            kernel_size=1,stride=1)
        
        # load weights
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def forward(self,x,h,w,threshold:float=None):
        x = self.conv1(x)
        cls,reg = self.conv2_cls(x),self.conv2_reg(x)
        cls = self.softmax(cls)
        
        ch,cw = cls.size()[-2:]

        # batch,k*2,ch,cw => batch,k,2,ch,cw => batch,k,ch,cw,2
        cls = cls.view(-1,self.anchor_size,2,ch,cw).permute(0,1,3,4,2)
        
        # batch,k*4,ch,cw => batch,k,4,ch,cw => batch,k,ch,cw,4
        reg = reg.view(-1,self.anchor_size,4,ch,cw).permute(0,1,3,4,2)
        
        # filter candidates using score threshold
        cls,bboxes = self.apply_nms(cls,reg,threshold,h,w)

        return cls,bboxes
    
    def apply_nms(self,cls:torch.Tensor,reg:torch.Tensor,threshold:float,h=int,w=int):

        if not threshold:
            threshold = self.threshold
        
        b,a,y,x = torch.where(cls[:,:,:,:,1]>=threshold)

        # filter preds using threshold
        reg = reg[b,a,y,x,:]
        cls = cls[b,a,y,x,1]

        # get selected anchors and scale it
        bboxes = self._get_anchors((a,y,x),h,w)

        # apply offset using regression to the bboxes
        bboxes = self._refine_anchors(bboxes,reg)

        # filter with NMS
        bboxes = self._xywh2xyxy(bboxes)
        pick = torchvision.ops.nms(bboxes,cls,self.iou_threshold)
        bboxes = bboxes[pick,:]
        cls = cls[pick]

        # get only N of bboxes
        return cls[:self._N],bboxes[:self._N,:]

    def _get_anchors(self,selected_indexes,h,w):
        bboxes = []
        offset_x = ((w-self.offset*2)%self.stride)/2 + self.offset
        offset_y = ((h-self.offset*2)%self.stride)/2 + self.offset
        a,y,x = selected_indexes
        for i in range(a.size()[0]):
            # TODO: check if w,h is same as x,y order with anchor selection(!?)

            cx = x[i]*self.stride+offset_x
            cy = y[i]*self.stride+offset_y
            #print(cx.view(1,1).size())
            #print(cy.view(1,1).size())
            #print("asd",a[i])
            #print(self.anchors[a[i]].view(1,2).size())

            bbox = torch.cat([cx.view(1,1),cy.view(1,1),self.anchors[a[i]].view(1,2)],dim=1)
            bboxes.append(bbox)
        return torch.cat(bboxes,dim=0)

    def _refine_anchors(self,anchors:torch.Tensor,reg:torch.Tensor):
        """Appling bounding box reggresion
        
        Arguments:
            anchors {torch.Tensor} -- N,4 shape (center_x,center_y,width,height)
            reg {torch.Tensor} -- N,4 shape
        
        Returns:
            torch.Tensor -- [N,4] shape tensor with (center_x,center_y,width,height)
        """
        
        """
            dx,xa,x': delta,anchor,ground truth
            dy,ya,y': delta,anchor,ground truth
            dw,wa,w': delta,anchor,ground truth
            dh,ha,h': delta,anchor,ground truth

            prediction
            x  = dx*wa+xa
            y  = dy*ha+ya
            w  = exp(dw)*wa
            h  = exp(wh)*ha

            ground truth
            x',y',w',h' 
        """
        x = reg[:,0] * anchors[:,2] + anchors[:,0]
        y = reg[:,1] * anchors[:,3] + anchors[:,1]
        w = torch.exp(reg[:,2])*anchors[:,2]
        h = torch.exp(reg[:,3])*anchors[:,3]
        
        return torch.stack([x,y,w,h]).t()

    def _xywh2xyxy(self,bboxes:torch.Tensor):
        x1 = bboxes[:,0] - bboxes[:,2]*0.5
        y1 = bboxes[:,1] - bboxes[:,3]*0.5
        x2 = bboxes[:,0] + bboxes[:,2]*0.5
        y2 = bboxes[:,1] + bboxes[:,3]*0.5
        return torch.stack([x1,y1,x2,y2]).t()
        

    def apply_threshold(self,cls:torch.Tensor,reg:torch.Tensor,threshold:float) -> tuple:
        threshold = self.threshold if isinstance(threshold,type(None)) else threshold
        
        b,a,y,x = torch.where(cls[:,:,:,:,1]>threshold)

        # filter preds using threshold
        reg = reg[b,a,y,x,:]
        cls = cls[b,a,y,x,1]
        selected_anchors = self.anchors[a]

if __name__ == '__main__':
    # TODO handle empty tensors
    import cv2,sys
    orig_img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2RGB)
    h,w,c = orig_img.shape
    print(orig_img.shape)
    
    img = preprocess(orig_img)

    fe = FeatureNetwork()
    
    rpn = RPN(threshold=0.2,N=4,backbone="vgg16")

    with torch.no_grad():
        x = fe(img)
        cls,bboxes = rpn(x,h,w)
    print("cls :",cls.size())
    #print("reg :",reg.size())
    bboxes = bboxes.numpy().tolist()
    orig_img = cv2.cvtColor(orig_img,cv2.COLOR_RGB2BGR)
    for x1,y1,x2,y2 in bboxes:
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(orig_img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow("",orig_img)
    cv2.waitKey(0)
