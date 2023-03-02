import importlib
import os
import os.path as osp
from copy import deepcopy

from utils.registry import MODEL_REGISTRY


model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(f))[0] for f in os.listdir(model_folder) if f.endswith('_model.py')]
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]

def build_model(opt):
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    return model
