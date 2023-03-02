import importlib
import os
from copy import deepcopy

from utils.registry import ARCH_REGISTRY


__all__ = ['build_network']

arch_folder = os.path.dirname(os.path.abspath(__file__))
arch_filenames = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(arch_folder) if f.endswith('_arch.py')]
# _arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    net = ARCH_REGISTRY.get(opt.pop('type'))(**opt)
    return net
