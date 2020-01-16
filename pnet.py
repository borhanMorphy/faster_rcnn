import torch.nn as nn

class Pnet(nn.Module):
    def __init__(self,threshold:float=0.6,iou_threshold:float=0.5):
        super(Pnet,self).__init__()
        self.threshold = threshold
        self.iou_threshold = iou_threshold


    def forward(self,x):
        
        # preprocess if needed

        # generate windows (?)

        # forward with conv net

        # filter with threshold
        
        # apply bbox regression

        # apply nms

    def bbox_regression(self,bboxes:torch.Tensor,reg:torch.Tensor):
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
