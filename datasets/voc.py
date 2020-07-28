from torchvision.datasets import VOCDetection
import numpy as np

class VOCDataset(VOCDetection):
    __label_mapper = {
        '__background__':0,
        'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
        'bottle': 7, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
        'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
        'motorbike': 14, 'person': 15, 'pottedplant': 16,
        'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
    }

    def __init__(self, *args, **kwargs):
        transforms = kwargs.pop('transforms', None)
        phase = kwargs.pop('phase','train')
        label_mapper = kwargs.pop('label_mapper', VOCDataset.__label_mapper)
        args = args if len(args) > 0 else ('./data',)
        assert phase in ["train","test","val"],"phase is invalid"
        kwargs['image_set'] = phase
        super(VOCDataset,self).__init__(*args, **kwargs)
        self._transforms = transforms
        self._label_mapper = label_mapper

    def __getitem__(self, idx):
        img,otargets = super().__getitem__(idx) # img: PIL.Image

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

class VOCDataset_train(VOCDataset):
    def __init__(self,*args,**kwargs):
        kwargs['phase'] = 'train'
        super(VOCDataset_train,self).__init__(*args,**kwargs)

class VOCDataset_val(VOCDataset):
    def __init__(self,*args,**kwargs):
        kwargs['phase'] = 'val'
        super(VOCDataset_val,self).__init__(*args,**kwargs)

class VOCDataset_test(VOCDataset):
    def __init__(self,*args,**kwargs):
        kwargs['phase'] = 'test'
        super(VOCDataset_test,self).__init__(*args,**kwargs)