from torch.utils.data import Dataset
from cv2 import cv2
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, label_mapper, phase:str='train', transforms=None):
        super(CustomDataset,self).__init__()
        self.phase = phase
        self._transforms = transforms
        self._label_mapper = label_mapper

    def __getitem__(self, idx):

        targets = {
            'boxes':[],
            'labels':[],
            'img_dims':None
        }
        img_size = (int(otargets['annotation']['size']['height']), int(otargets['annotation']['size']['width']))
        for target in otargets['annotation']['object']:
            if target['name'] not in self._label_mapper: continue

            targets['boxes'].append(
                [target['bndbox']['xmin'],target['bndbox']['ymin'],target['bndbox']['xmax'],target['bndbox']['ymax']]
            )

            targets['labels'].append( self._label_mapper[target['name']] )

        targets['boxes'] = np.array(targets['boxes'], dtype=np.float32)
        targets['labels'] = np.array(targets['labels'], dtype=np.int32)
        targets['img_dims'] = np.array(img_size, dtype=np.int32)


        if self._transforms is not None:
            img,targets = self._transforms(img, targets=targets)

        return img,targets

    @staticmethod
    def _load_img(img_path:str):
        return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)