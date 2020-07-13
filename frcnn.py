import torch.nn as nn
import torch


class FasterRCNN(nn.Module):
    def __init__(self, backbone:nn.Module, )