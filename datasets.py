import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from make_noise import make_noise

def get_planetoid_dataset(path, name, normalize_features=False, transform=None, type=None, rate=None, seed=0,split_type='full'):
    
    path = osp.join(path, name)
    if split_type == 'random':
        dataset = Planetoid(path, name,split=split_type,num_train_per_class=150, num_val=500, num_test =1000)
    elif split_type == "full":
        dataset = Planetoid(path, name,split=split_type)
    elif split_type == "public":    
        dataset = Planetoid(path, name,split=split_type)

    clean_label = make_noise(dataset, type, rate, seed)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset, clean_label

