import torch
import sys
import numpy as np
from collections import OrderedDict

mtcnn_weight_path = "/home/borhanmorphy/localwork/mtcnn/mtcnn/data/mtcnn_weights.npy"
weights = np.load(mtcnn_weight_path,allow_pickle=True).tolist()

layers = [
    'conv1.weight','conv1.bias','prelu1.weight',
    'conv2.weight','conv2.bias','prelu2.weight',
    'conv3.weight','conv3.bias','prelu3.weight',
    'conv4a.weight','conv4a.bias',
    'conv4b.weight','conv4b.bias'
]

state_dict = OrderedDict()

for values,layer in zip(weights.get("pnet"),layers):
    # convert kernel,kernel,channel,fmap => fmap,channel,kernel,kernel
    if len(values.shape) == 4:
        values = np.transpose(values,(3,2,1,0))
    elif len(values.shape) == 3:
        values = np.squeeze(values)
    state_dict[layer] = torch.from_numpy(values)

torch.save(state_dict,"pnet.pth")

