import importlib
import os
from copy import deepcopy

import torch.utils.data as data

from utils.registry import DATASET_REGISTRY


data_folder = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(data_folder) if f.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]

def build_dataset(opt):
    opt = deepcopy(opt)
    dataset = DATASET_REGISTRY.get(opt['type'])(opt)
    return dataset

def build_dataloader(dataset, dataset_opt, num_gpu=1):
    phase = dataset_opt['phase']
    # non-distributed training
    if phase == 'train':
        multiplier = 1 if num_gpu == 0 else num_gpu
        batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
        num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
    elif phase in ['val', 'test']: # validation
        dataloader_args = dict(dataset=dataset,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}')

    return data.DataLoader(**dataloader_args)
