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
    split_dataset,
    read_csv,
    split
)
import os
import argparse
import json
import time

def parse_data(root_path:str):
    ann_path = os.path.join(root_path,'train.csv')
    data_path = os.path.join(root_path,'train')
    
    mapper = {}

    rows,headers = read_csv(ann_path)
    boxes = rows[headers.index('bbox')]
    img_ids = rows[headers.index('image_id')]

    for img_id,box in zip(img_ids,boxes):
        x,y,w,h = [float(b.strip()) for b in box[1:-1].split(',')]
        box = [x,y,x+w,y+h]
        if img_id not in mapper:
            mapper[img_id] = [box]
        else:
            mapper[img_id].append(box)

    ids = []
    labels = []
    label_mapper = {'__background__':0, 'wheed':1}
    for k,v in mapper.items():
        ids.append(os.path.join(data_path,k+'.jpg'))
        labels.append({'boxes': v, 'labels':[1]*len(v)})

    return ids,labels,label_mapper

def parse_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument('--root-path', '-r', type=str, required=True)
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

    ids,labels,label_mapper = parse_data(args.root_path)

    train_transforms = TrainTransforms((1024,1024))
    val_transforms = TestTransforms((1024,1024))
    batch_size = args.batch_size
    epochs = args.epochs

    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_classes = 2

    val_data,train_data = split(list(zip(ids,labels)), ratio=args.val_ratio)

    val_ids,val_labels = zip(*val_data)
    train_ids,train_labels = zip(*train_data)

    ds_train = ds_factory("custom", label_mapper, train_ids, train_labels, phase='train', transforms=train_transforms)
    ds_val = ds_factory("custom", label_mapper, val_ids, val_labels, phase='val', transforms=val_transforms)

    dl_train = generate_dl(ds_train, batch_size=batch_size)
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
        torch.save(model.state_dict(), f"./custom_model_epoch_{epoch+1}.pth")

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
    head_predictions = [pred[:,:5] for pred in head_predictions]
    head_ground_truths = [pred[:,:4] for pred in head_ground_truths]

    AP50 = calculate_AP(head_predictions, head_ground_truths, iou_threshold=0.5)
    AP75 = calculate_AP(head_predictions, head_ground_truths, iou_threshold=0.75)
    AP90 = calculate_AP(head_predictions, head_ground_truths, iou_threshold=0.90)
    AP = (AP50 + AP75 + AP90) / 3
    means = caclulate_means(all_losses)

    print(f"--validation results for epoch {epoch+1} --")
    print(f"RPN mean recall at iou thresholds are:")
    for iou_threshold,rpn_recall in zip(iou_thresholds.cpu().numpy(),rpn_recalls.cpu().numpy()*100):
        print(f"IoU={iou_threshold:.02f} recall={int(rpn_recall)}")
    print(f"HEAD AP IoU=.5 :{AP50.item()*100:.02f}")
    print(f"HEAD AP IoU=.75 :{AP75.item()*100:.02f}")
    print(f"HEAD AP IoU=.90 :{AP90.item()*100:.02f}")
    print(f"HEAD AP IoU=.5:.95 :{AP.item()*100:.02f}")

    for k,v in means.items():
        print(f"{k}: {v:.4f}")
    print("--------------------------------------------")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()
    main(args)