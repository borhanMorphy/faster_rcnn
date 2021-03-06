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
import argparse
import json

def parse_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--batch-size', '-bs', type=int, default=1)
    ap.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    ap.add_argument('--momentum', '-m', type=float, default=.9)
    ap.add_argument('--epochs', '-e', type=int, default=10)
    ap.add_argument('--weight-decay', '-wd', type=float, default=5e-4)

    ap.add_argument('--val-ratio', '-vr', type=float, default=1e-1)
    ap.add_argument('--verbose', '-vb', type=int, default=50) # every 50 forward show logs

    return ap.parse_args()

def load_latest_checkpoint(model):
    checkpoints = [f_name for f_name in os.listdir() if f_name.endswith('.pth')]
    for checkpoint in sorted(checkpoints, reverse=True):
        print(f"found checkpoint {checkpoint}")
        model.load_state_dict( torch.load(checkpoint, map_location='cpu') )
        return

def main(args):
    print(json.dumps(vars(args),sort_keys=False,indent=4))
    train_transforms = TrainTransforms()
    val_transforms = TestTransforms()
    batch_size = args.batch_size
    epochs = args.epochs

    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_classes = 21

    ds_train = ds_factory("VOC_train", transforms=train_transforms, download=not os.path.isfile('./data/VOCtrainval_11-May-2012.tar'))
    dl_train = generate_dl(ds_train, batch_size=batch_size)

    ds_val = ds_factory("VOC_val", transforms=val_transforms, download=not os.path.isfile('./data/VOCtrainval_11-May-2012.tar'))
    ds_val = reduce_dataset(ds_val, ratio=args.val_ratio)
    dl_val = generate_dl(ds_val, batch_size=batch_size)

    backbone = models.mobilenet_v2(pretrained=True).features
    backbone.output_channels = 1280

    model = FasterRCNN(backbone,num_classes)

    load_latest_checkpoint(model)

    model.to('cuda')

    verbose = int(args.verbose/batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
        momentum=momentum, weight_decay=weight_decay)

    max_iter_count = int(len(ds_train)/batch_size)

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

def validation_loop(model, dl, batch_size:int, epoch:int):
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

    mAP = calculate_mAP(head_predictions, head_ground_truths, model.head.num_classes, iou_threshold=0.5)
    AP = calculate_AP([pred[:,:5] for pred in head_predictions], [gt[:,:4] for gt in head_ground_truths], iou_threshold=0.5)

    means = caclulate_means(all_losses)

    print(f"--validation results for epoch {epoch+1} --")
    print(f"RPN mean recall at iou thresholds are:")
    for iou_threshold,rpn_recall in zip(iou_thresholds.cpu().numpy(),rpn_recalls.cpu().numpy()*100):
        print(f"IoU={iou_threshold:.02f} recall={int(rpn_recall)}")
    print(f"HEAD AP objectness score={AP.item()*100:.02f} at IoU=0.5")
    print(f"HEAD mAP score={mAP.item()*100:.02f} at IoU=0.5")
    for k,v in means.items():
        print(f"{k}: {v:.4f}")
    print("--------------------------------------------")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()
    main(args)
