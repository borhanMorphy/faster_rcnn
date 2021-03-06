import torch
from torch.utils.data import DataLoader
import random

def reduce_dataset(ds,ratio=0.1):
    return split_dataset(ds,ratio=ratio)[0]

def split_dataset(ds, ratio=0.1):
    size = int(len(ds)*ratio)
    rest = len(ds)-size
    return torch.utils.data.random_split(ds, [size,rest])

def split(data, ratio=0.1):
    random.shuffle(data)

    total = len(data)
    cut = int(total * ratio)
    return data[:cut],data[cut:]

def custom_collate_fn(batch):
    images,targets = zip(*batch)
    images = torch.cat(images,dim=0)
    return images,targets

def generate_dl(ds, batch_size:int=1, collate_fn=custom_collate_fn,
        num_workers:int=1, pin_memory:bool=True, **kwargs):

    return DataLoader(ds, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=True, **kwargs)