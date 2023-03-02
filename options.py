import os
import yaml
import argparse
from collections import OrderedDict


def _ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(yamlfile):
    if os.path.isfile(yamlfile):
        with open(yamlfile, 'r') as f:
            return yaml.load(f, Loader=_ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=_ordered_yaml()[0])


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser(description='Super Resolution Video')
    parser.add_argument('--opt', type=str, default='./config/train/ESPCN/train_DIV2K_x4.yml')

    args = parser.parse_args()
    # opt = vars(args)
    opt = yaml_load(args.opt)
    opt['is_train'] = is_train
    # opt['auto_resume'] = args.auto_resume

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('gt_dir') is not None:
            dataset['gt_dir'] = os.path.expanduser(dataset['gt_dir'])
        if dataset.get('lq_dir') is not None:
            dataset['lq_dir'] = os.path.expanduser(dataset['lq_dir'])

    return opt
