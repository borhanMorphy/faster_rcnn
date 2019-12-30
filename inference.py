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
        self.features = torchvision.models.alexnet(pretrained=True).features

    def forward(self,x):
        return self.features(x)

class RPN(torch.nn.Module):
    def __init__(self):
        super(RPN,self).__init__()
        self.anchor_ratios = torch.tensor([[1,1]],dtype=torch.float32)
        self.anchor_scales = torch.tensor([128],dtype=torch.float32)

        self.conv1 = torch.nn.Conv2d(in_channels=256,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.conv2_cls = torch.nn.Conv2d(in_channels=1,out_channels=2,kernel_size=1,stride=1)
        self.conv2_reg = torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=1,stride=1)

    def forward(self,x):
        x = self.conv1(x)
        return self.conv2_cls(x),self.conv2_reg(x)



if __name__ == '__main__':
    import cv2,sys
    img = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2RGB)
    img = preprocess(img)
    fe = FeatureNetwork()
    rpn = RPN()

    with torch.no_grad():
        x = fe(img)
        cls,reg = rpn(x)
    print("cls :",cls.size())
    print("reg :",reg.size())