import torchvision.models as models
from neuralnets import RPN
from datasets import factory as ds_factory
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict,Tuple,List
import torch.nn.functional as F
from utils.metrics import (
    caclulate_means,
    roi_recalls,
    calculate_mAP,
    calculate_AP
)
from utils import reduce_dataset
from transforms.preprocess import TrainTransforms,TestTransforms
import os

def load_latest_checkpoint(model):
    checkpoints = [f_name for f_name in os.listdir() if f_name.endswith('.pth') and 'rpn' in f_name]
    for checkpoint in sorted(checkpoints, reverse=True):
        print(f"found checkpoint {checkpoint}")
        model.load_state_dict( torch.load(checkpoint, map_location='cpu') )
        return

def custom_collate_fn(batch):
    images,targets = zip(*batch)
    return images,targets

def generate_dl(ds, batch_size:int=1, collate_fn=custom_collate_fn,
        num_workers:int=1, pin_memory:bool=True, **kwargs):

    return DataLoader(ds, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=True, **kwargs)

def main():
    small_dim_size = 600
    train_transforms = TrainTransforms(small_dim_size=small_dim_size)
    val_trainsforms = TestTransforms(small_dim_size=small_dim_size)

    batch_size = 3
    epochs = 1

    # !defined in the paper
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    total_iter_size = 60000
    features = 1280

    ds_train = ds_factory("VOC_train", transforms=train_transforms, download=not os.path.isfile('./data/VOCtrainval_11-May-2012.tar'))
    dl_train = generate_dl(ds_train, batch_size=batch_size)

    ds_val = ds_factory("VOC_val", transforms=val_trainsforms, download=not os.path.isfile('./data/VOCtrainval_11-May-2012.tar'))
    ds_val = reduce_dataset(ds_val, ratio=0.1)
    dl_val = generate_dl(ds_val, batch_size=batch_size)

    backbone = models.mobilenet_v2(pretrained=True).features
    backbone.output_channels = features

    model = RPN(backbone)

    load_latest_checkpoint(model)

    model.to('cuda')

    verbose = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
        momentum=momentum, weight_decay=weight_decay)

    max_iter_count = int(len(ds_train)/batch_size)
    # ! set because of the paper
    epochs = int(total_iter_size / max_iter_count)

    for epoch in range(epochs):
        # start validation
        validation_loop(model, dl_val, batch_size, epoch)

        # start training
        train_loop(model, dl_train, batch_size, epoch, epochs, optimizer, verbose, max_iter_count)

        # save checkpoint
        print("saving checkpoint...")
        torch.save(model.state_dict(), f"./rpn_checkpoint_epoch_{epoch+1}.pth")

def move_to_gpu(batch:List[torch.Tensor], targets:List[Dict[str,torch.Tensor]]):
    for i in range(len(batch)):
        batch[i] = batch[i].cuda()
        targets[i]['boxes'] = targets[i]['boxes'].cuda()
        targets[i]['labels'] = targets[i]['labels'].cuda()
    return batch,targets

def train_loop(model, dl, batch_size:int, epoch, epochs, optimizer, verbose, max_iter_count):
    running_metrics = []
    print(f"running epoch [{epoch+1}/{epochs}]")
    model.train()
    for iter_count,(batch,targets) in enumerate(dl):
        batch,targets = move_to_gpu(batch,targets)
        optimizer.zero_grad()
        metrics = model.training_step(batch, targets)
        metrics['loss'].backward()
        optimizer.step()

        metrics['loss'] = metrics['loss'].item()

        running_metrics.append(metrics)
        if (iter_count+1) % verbose == 0:
            means = caclulate_means(running_metrics)
            running_metrics = []
            log = []
            for k,v in means.items():
                log.append(f"{k}: {v:.04f}")
            log = "\t".join(log)
            log += f"\titer[{iter_count}/{max_iter_count}]"
            print(log)

def validation_loop(model, dl, batch_size:int, epoch):
    # start validation
    total_val_iter = int(len(dl.dataset) / batch_size)
    model.eval()
    print("running validation...")
    all_detections = []
    all_losses = []
    for batch,targets in tqdm(dl, total=total_val_iter):
        batch,targets = move_to_gpu(batch,targets)

        detections,losses = model.validation_step(batch, targets)
        #draw_it(batch[0],detections['predictions'],detections['ground_truths'])
        all_detections.append(detections)
        all_losses.append(losses)

    # evalute RPN
    iou_thresholds = torch.arange(0.5, 1.0, 0.05)
    rpn_predictions = []
    rpn_ground_truths = []
    for dets in all_detections:
        rpn_predictions += dets['predictions']
        rpn_ground_truths += dets['ground_truths']

    rpn_recalls = roi_recalls(rpn_predictions, rpn_ground_truths, iou_thresholds=iou_thresholds)
    ap = calculate_AP(rpn_predictions,rpn_ground_truths,iou_threshold=0.5)
    means = caclulate_means(all_losses)

    print(f"--validation results for epoch {epoch+1} --")
    print(f"RPN mean recall at iou thresholds are:")
    for iou_threshold,rpn_recall in zip(iou_thresholds.cpu().numpy(),rpn_recalls.cpu().numpy()*100):
        print(f"IoU={iou_threshold:.02f} recall={int(rpn_recall)}")
    print(f"AP score: {100*ap:.02f}")
    for k,v in means.items():
        print(f"{k}: {v:.4f}")
    print("--------------------------------------------")

def draw_it(batch,detections,gt_boxes):
    from torchvision.ops import boxes as box_utils
    ious = box_utils.box_iou(detections[0][:,:4],gt_boxes[0])
    vals,ids = ious.max(dim=0)
    print(vals)
    from utils.data import tensor2img
    img = tensor2img(batch)[0]
    boxes = detections[0][ids].cpu().long().numpy()
    gt_boxes = gt_boxes[0].cpu().long().numpy()
    from cv2 import cv2
    for x1,y1,x2,y2 in boxes[:,:4]:
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
    for x1,y1,x2,y2 in gt_boxes:
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    
    cv2.imshow("",img)
    cv2.waitKey(0)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
