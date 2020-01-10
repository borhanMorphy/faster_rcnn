import torch
import torchvision
import numpy as np


def preprocess(x:np.ndarray):
    x = x.astype(np.float32)
    x /= 255.0
    x = np.transpose(x,(2,0,1))
    x = np.expand_dims(x,axis=0)
    return torch.from_numpy(x)


class FeatureNetwork(torch.nn.Module):
    def __init__(self):
        super(FeatureNetwork,self).__init__()
        self.features = torchvision.models.alexnet(pretrained=True).features[:-1]

    def forward(self,x):
        return self.features(x)

class RPN(torch.nn.Module):
    def __init__(self,threshold:float=0.4,N:int=50,iou_threshold:float=0.6):
        super(RPN,self).__init__()
        anchor_ratios = torch.tensor([[1,1]],dtype=torch.float32)
        anchor_scales = [64,128]
        
        self.r = 16
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

    def forward(self,x,threshold:float=None):
        x = self.conv1(x)
        cls,reg = self.conv2_cls(x),self.conv2_reg(x)
        ch,cw = cls.size()[-2:]

        # batch,k*2,ch,cw => batch,k,2,ch,cw => batch,k,ch,cw,2
        cls = cls.view(-1,self.anchor_size,2,ch,cw).permute(0,1,3,4,2)
        
        # batch,k*4,ch,cw => batch,k,4,ch,cw => batch,k,ch,cw,4
        reg = reg.view(-1,self.anchor_size,4,ch,cw).permute(0,1,3,4,2)
        

        print("anchors: ",selected_anchors)
        print("anchors shape: ",selected_anchors.size())
        print("regression result: ",reg.size())
        print("classification result: ",cls.size())
        """
        for reg,score in zip(reg,cls[:,:,1]): # for every batch, apply NMS
            bboxes = reg*
        """
        return cls,reg

    def apply_threshold(self,cls:torch.Tensor,reg:torch.Tensor,threshold:float) -> tuple:
        threshold = self.threshold if isinstance(threshold,type(None)) else threshold
        
        b,a,y,x = torch.where(cls[:,:,:,:,1]>threshold)

        # filter preds using threshold
        reg = reg[b,a,y,x,:]
        cls = cls[b,a,y,x,1]
        selected_anchors = self.anchors[a]
        


if __name__ == '__main__':
    import cv2,sys
    img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2RGB)
    print(img.shape)
    img = preprocess(img)
    fe = FeatureNetwork()
    
    rpn = RPN()

    with torch.no_grad():
        x = fe(img)
        cls,reg = rpn(x)
    print("cls :",cls.size())
    print("reg :",reg.size())