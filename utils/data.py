import torch
from cv2 import cv2
import numpy as np
from typing import List,Dict
import csv


def load_data(img_path:str):
    oimg = cv2.imread(img_path)
    data = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
    return (torch.from_numpy(data).float() / 255).permute(2,0,1).unsqueeze(0),oimg


def tensor2img(batch):
    imgs = (batch.permute(0,2,3,1).cpu() * 255).numpy().astype(np.uint8)
    return [cv2.cvtColor(img,cv2.COLOR_RGB2BGR) for img in imgs]


def move_to_gpu(batch:torch.Tensor, targets:List[Dict[str,torch.Tensor]]):
    for i in range(len(targets)):
        targets[i]['boxes'] = targets[i]['boxes'].cuda()
        targets[i]['labels'] = targets[i]['labels'].cuda()
    batch = batch.cuda()
    return batch,targets


def read_csv(file_path:str):
    rows = {}
    headers = []
    with open(file_path,"r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for i,r in enumerate(row):
                    rows[i] = []
                    headers.append(r)
                line_count += 1
            else:
                for i,r in enumerate(row):
                    rows[i].append(r)
                line_count += 1
    return rows,headers