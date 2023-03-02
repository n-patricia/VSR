import importlib
import os
import os.path as osp
from copy import deepcopy

from utils.registry import LOSS_REGISTRY

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(f))[0] for f in os.listdir(loss_folder) if f.endswith('_loss.py')]
_model_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]

def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    return loss
