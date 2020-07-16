import torchvision.models as models
from rpn import RPN
from datasets import factory as ds_factory
import torch
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict
import torch.nn.functional as F
from utils.metrics import caclulate_means,roi_recalls
from transforms import TrainTransforms

def custom_collate_fn(batch):
    batch,targets = zip(*batch)
    return torch.cat(batch,dim=0),targets

def generate_dl(ds, batch_size:int=1, collate_fn=custom_collate_fn,
        num_workers:int=1, pin_memory:bool=True, **kwargs):

    return DataLoader(ds, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=True, **kwargs)

def reduce_dataset(ds,ratio=0.1):
    size = int(len(ds)*ratio)
    rest = len(ds)-size
    return torch.utils.data.random_split(ds, [size,rest])[0]

def main():
    train_transforms = TrainTransforms()

    debug = False # TODO add debug
    batch_size = 1
    epochs = 1

    # !defined in the paper
    learning_rate = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    total_iter_size = 60000

    ds_train = ds_factory("VOC_train", transforms=train_transforms, download=False)
    dl_train = generate_dl(ds_train, batch_size=batch_size)

    ds_val = ds_factory("VOC_val", transforms=train_transforms, download=False)
    ds_val = reduce_dataset(ds_val, ratio=0.1)
    dl_val = generate_dl(ds_val, batch_size=batch_size)

    backbone = models.vgg16(pretrained=True).features[:-1]

    rpn = RPN(backbone, features=512, n=3, effective_stride=16,
        conf_threshold=0.0, iou_threshold=0.7, keep_n=2000)
    #rpn = RPN(backbone, features=512, n=3, effective_stride=16) # for vgg16
    rpn.debug = debug
    rpn.to('cuda')

    verbose = 100
    optimizer = torch.optim.SGD(rpn.parameters(), lr=learning_rate,
        momentum=momentum, weight_decay=weight_decay)

    max_iter_count = int(len(ds_train)/batch_size)
    # ! set because of the paper
    epochs = int(total_iter_size / max_iter_count)

    for epoch in range(epochs):
        # start training
        train_loop(rpn, dl_train, batch_size, epoch, epochs, optimizer, verbose, max_iter_count)

        # save checkpoitn
        torch.save(rpn.state_dict(), f"./rpn_epoch_{epoch+1}.pth")

        # start validation
        validation_loop(rpn, dl_val, batch_size, epoch)

def train_loop(model, dl, batch_size:int, epoch, epochs, optimizer, verbose, max_iter_count):
    running_metrics = []
    print(f"running epoch [{epoch+1}/{epochs}]")
    model.train()
    for iter_count,(batch,targets) in enumerate(dl):

        optimizer.zero_grad()
        metrics = model.training_step(batch.cuda(), targets)
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
    predictions = []
    ground_truths = []
    for batch,targets in tqdm(dl, total=total_val_iter):
        with torch.no_grad():
            preds = model.test_step(batch.cuda(), targets)
        ground_truths.append(targets[0]['boxes'].cpu())
        predictions.append(preds[0].cpu())

    iou_thresholds = torch.arange(0.3, 1.0, 0.05, device=predictions[0].device)
    recalls = roi_recalls(predictions, ground_truths, iou_thresholds=iou_thresholds)
    print(f"validation results for epoch {epoch+1}, RPN mean recall at iou thresholds {iou_thresholds} is: {int(recalls.mean()*100)}")

if __name__ == "__main__":
    main()
