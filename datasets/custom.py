from torch.utils.data import Dataset
from cv2 import cv2
import numpy as np
from typing import Dict,List
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self, label_mapper:Dict[int,str],
            ids:List, labels:List, phase:str='train', transforms=None):
        super(CustomDataset,self).__init__()
        assert len(ids) == len(labels),"ids and labels length must match"
        self.phase = phase
        self.ids = ids
        self.labels = labels
        self._transforms = transforms
        self._label_mapper = label_mapper

    def __getitem__(self, idx:int):
        targets = self.labels[idx]
        img = self._load_img(self.ids[idx]) # PIL image
        assert 'boxes' in targets
        assert 'labels' in targets
        
        targets['boxes'] = np.array(targets['boxes'], dtype=np.float32)
        targets['labels'] = np.array(targets['labels'], dtype=np.int64)

        if self._transforms is not None:
            img,targets = self._transforms(img, targets=targets)

        return img,targets

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _load_img(img_path:str):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)