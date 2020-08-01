import torchvision.models as models
from neuralnets import FasterRCNN
from datasets import factory as ds_factory
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict
import torch.nn.functional as F
from utils.metrics import (
    caclulate_means,
    roi_recalls,
    calculate_mAP,
    calculate_AP
)
from transforms import TrainTransforms,TestTransforms
from utils import (
    move_to_gpu,
    generate_dl,
    reduce_dataset
)
import os

def load_latest_checkpoint(model):
    checkpoints = [f_name for f_name in os.listdir() if f_name.endswith('.pth')]
    for checkpoint in sorted(checkpoints, reverse=True):
        print(f"found checkpoint {checkpoint}")
        model.load_state_dict( torch.load(checkpoint, map_location='cpu') )
        return

def main():
    small_dim_size = 800
    train_transforms = TrainTransforms(small_dim_size=small_dim_size)
    val_transforms = TestTransforms(small_dim_size=small_dim_size)
    batch_size = 1
    epochs = 1

    # !defined in the paper
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    total_iter_size = 60000 / batch_size
    num_classes = 21

    ds_train = ds_factory("VOC_train", transforms=train_transforms, download=not os.path.isfile('./data/VOCtrainval_11-May-2012.tar'))
    dl_train = generate_dl(ds_train, batch_size=batch_size)

    ds_val = ds_factory("VOC_val", transforms=val_transforms, download=not os.path.isfile('./data/VOCtrainval_11-May-2012.tar'))
    ds_val = reduce_dataset(ds_val, ratio=0.1)
    dl_val = generate_dl(ds_val, batch_size=batch_size)

    backbone = models.mobilenet_v2(pretrained=True).features
    backbone.output_channels = 1280

    model = FasterRCNN(backbone,num_classes)

    load_latest_checkpoint(model)

    model.to('cuda')

    verbose = int(50/batch_size)
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
        torch.save(model.state_dict(), f"./faster_rcnn_epoch_{epoch+1}.pth")

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
        
        all_losses.append(losses)
        all_detections.append(detections)

    # evalute RPN
    iou_thresholds = torch.arange(0.5, 1.0, 0.05)
    rpn_predictions = []
    rpn_ground_truths = []
    for dets in all_detections:
        rpn_predictions += dets['rpn']['predictions']
        rpn_ground_truths += dets['rpn']['ground_truths']

    rpn_recalls = roi_recalls(rpn_predictions, rpn_ground_truths, iou_thresholds=iou_thresholds)

    # evalute FastRCNN
    head_predictions = []
    head_ground_truths = []
    for dets in all_detections:
        head_predictions += dets['head']['predictions']
        head_ground_truths += dets['head']['ground_truths']

    AP = calculate_AP(head_predictions, head_ground_truths, iou_threshold=0.5)

    means = caclulate_means(all_losses)

    print(f"--validation results for epoch {epoch+1} --")
    print(f"RPN mean recall at iou thresholds are:")
    for iou_threshold,rpn_recall in zip(iou_thresholds.cpu().numpy(),rpn_recalls.cpu().numpy()*100):
        print(f"IoU={iou_threshold:.02f} recall={int(rpn_recall)}")
    print(f"HEAD AP score={AP.item()*100:.02f} at IoU=0.5")
    for k,v in means.items():
        print(f"{k}: {v:.4f}")
    print("--------------------------------------------")

def tensor2img(batch):
    imgs = (batch.permute(0,2,3,1).cpu() * 255).numpy().astype(np.uint8)
    return [cv2.cvtColor(img,cv2.COLOR_RGB2BGR) for img in imgs]

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
