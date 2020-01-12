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
        self.features = torchvision.models.alexnet(pretrained=True).features[:-1]

    def forward(self,x,**kwargs):
        return self.features(x,**kwargs)

class RPN(torch.nn.Module):
    def __init__(self,threshold:float=0.7,N:int=50,iou_threshold:float=0.6):
        super(RPN,self).__init__()
        anchor_ratios = torch.tensor([[1,1],[1,2]],dtype=torch.float32) # x:y
        anchor_scales = [64,128]
        
        self.stride = 16         # for alexnet
        self.offset = 17.5       # for alexnet
        
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
            in_channels=256,out_channels=256,
            kernel_size=3,stride=1,padding=1)
        
        self.conv2_cls = torch.nn.Conv2d(
            in_channels=256,out_channels=2*self.anchor_size,
            kernel_size=1,stride=1)
        
        self.conv2_reg = torch.nn.Conv2d(
            in_channels=256,out_channels=4*self.anchor_size,
            kernel_size=1,stride=1)

    def forward(self,x,h,w,threshold:float=None):
        x = self.conv1(x)
        cls,reg = self.conv2_cls(x),self.conv2_reg(x)
        ch,cw = cls.size()[-2:]

        # batch,k*2,ch,cw => batch,k,2,ch,cw => batch,k,ch,cw,2
        cls = cls.view(-1,self.anchor_size,2,ch,cw).permute(0,1,3,4,2)
        
        # batch,k*4,ch,cw => batch,k,4,ch,cw => batch,k,ch,cw,4
        reg = reg.view(-1,self.anchor_size,4,ch,cw).permute(0,1,3,4,2)
        
        # filter candidates using score threshold
        cls,reg,bboxes = self.apply_nms(cls,reg,threshold,h,w)

        """
        for reg,score in zip(reg,cls[:,:,1]): # for every batch, apply NMS
            bboxes = reg*
        """
        return cls,reg,bboxes
    
    def apply_nms(self,cls:torch.Tensor,reg:torch.Tensor,threshold:float,h=int,w=int):

        if not threshold:
            threshold = self.threshold
        
        b,a,y,x = torch.where(cls[:,:,:,:,1]>=threshold)

        # filter preds using threshold
        reg = reg[b,a,y,x,:]
        cls = cls[b,a,y,x,1]

        # get selected anchors and scale it
        bboxes = self._get_refined_anchors((a,y,x),h,w)

        # filter with NMS
        print("?",bboxes.size())
        return cls,reg,bboxes[:self._N,:]

    def _get_refined_anchors(self,selected_indexes,h,w):
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


if __name__ == '__main__':
    import cv2,sys
    orig_img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2RGB)
    h,w,c = orig_img.shape
    print(orig_img.shape)
    
    img = preprocess(orig_img)

    fe = FeatureNetwork()
    
    rpn = RPN(threshold=0.2,N=4)

    with torch.no_grad():
        x = fe(img)
        cls,reg,bboxes = rpn(x,h,w)
    print("cls :",cls.size())
    print("reg :",reg.size())
    bboxes = bboxes.numpy().tolist()
    orig_img = cv2.cvtColor(orig_img,cv2.COLOR_RGB2BGR)
    for cx,cy,w,h in bboxes:
        x1,y1 = int(cx-w/2),(int(cy-h/2))
        x2,y2 = int(cx+w/2),(int(cy+h/2))
        cv2.rectangle(orig_img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow("",orig_img)
    cv2.waitKey(0)
