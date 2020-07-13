from .voc import VOCDataset_test,VOCDataset_train,VOCDataset_val

__ds_mapper__ = {
    'VOC_train': {
        'cls': VOCDataset_train,
        'args':('./data',),
        'kwargs':{}
    },
    'VOC_val': {
        'cls': VOCDataset_val,
        'args':('./data',),
        'kwargs':{}
    },
    'VOC_test': {
        'cls': VOCDataset_test,
        'args':('./data',),
        'kwargs':{}
    }
}

def factory(dataset_name:str, **configs):
    assert dataset_name in __ds_mapper__

    cls = __ds_mapper__[dataset_name]['cls']
    args = __ds_mapper__[dataset_name]['args']
    kwargs = __ds_mapper__[dataset_name]['kwargs']
    kwargs.update(configs)

    return cls(*args,**kwargs)
