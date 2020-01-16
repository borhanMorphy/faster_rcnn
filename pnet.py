import torch.nn as nn
import torch

class Pnet(nn.Module):
    def __init__(self,threshold:float=0.6,iou_threshold:float=0.5):
        super(Pnet,self).__init__()
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.stride = 2
        self.offset = 6
        self.window = torch.tensor([12,12],dtype=torch.float32)

    def forward(self,x):
        pass
        
        # preprocess if needed

        # generate windows (?)

        # forward with conv net

        # filter with threshold
        
        # apply bbox regression

        # apply nms

    def _bbox_regression(self,bboxes:torch.Tensor,reg:torch.Tensor):
        """bounding box regression
        
        Arguments:
            bboxes {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
            reg {torch.Tensor} -- [N,4] with order of x1,y1,x2,y2
        
        Returns:
            torch.Tensor -- calibrated bounding boxes
        """
        
        w = bboxes[:,2]-bboxes[:,0]
        h = bboxes[:,3]-bboxes[:,1]

        x1 = bboxes[:,0] + reg[:,0]*w
        y1 = bboxes[:,1] + reg[:,1]*h
        x2 = bboxes[:,2] + reg[:,2]*w
        y2 = bboxes[:,3] + reg[:,3]*h

        return torch.stack([x1,y1,x2,y2]).t()

    def _get_windows(self,selected_indexes,h,w):
        bboxes = []
        
        offset_x = ((w-self.offset*2)%self.stride)/2 + self.offset
        offset_y = ((h-self.offset*2)%self.stride)/2 + self.offset
        
        y,x = selected_indexes
        
        cx = x * self.stride+offset_x
        cy = y * self.stride+offset_y
        bbox = torch.cat([cx.view(-1,1),cy.view(-1,1),self.window.view(-1,2)],dim=1)
        return torch.cat([bbox],dim=0)


if __name__ == '__main__':
    import cv2,sys
    img = cv2.imread(sys.argv[1])
    model = Pnet()
    h,w = img.shape[:2]
    y = torch.tensor([0,1]).t()
    x = torch.tensor([0,1]).t()
    print(model._get_windows((y,x),h,w))


    