import torch
import sys
import numpy as np
from collections import OrderedDict

mtcnn_weight_path = "../mtcnn/mtcnn/data/mtcnn_weights.npy"
weights = np.load(mtcnn_weight_path,allow_pickle=True).tolist()

layers = [
    'conv1.weight','conv1.bias','prelu1.weight',
    'conv2.weight','conv2.bias','prelu2.weight',
    'conv3.weight','conv3.bias','prelu3.weight',
    'linear4.weight','linear4.bias','prelu4.weight',
    'linear5a.weight','linear5a.bias',
    'linear5b.weight','linear5b.bias'
]

state_dict = OrderedDict()

for values,layer in zip(weights.get("rnet"),layers):
    # convert kernel,kernel,channel,fmap => fmap,channel,kernel,kernel
    if len(values.shape) == 4:
        values = np.transpose(values,(3,2,1,0))
    elif len(values.shape) == 3:
        values = np.squeeze(values)
    elif len(values.shape) == 2:
        values = np.transpose(values,(1,0))
    print(layer,"  ",values.shape)
    state_dict[layer] = torch.from_numpy(values)

torch.save(state_dict,"rnet.pth")

