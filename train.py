import os
import os.path as osp
import math
import time
from datetime import date
import logging

import torch

from data import build_dataset, build_dataloader
from models import build_model
from options import parse_options
from utils import get_logger, init_tb_logger


def get_device():
    device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
    return device


def create_train_val_dataloader(opt):
    train_loader, val_loader = None, None
    total_epochs = 100
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_opt['is_train'] = opt['is_train']
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'])
            num_iter_per_epoch = math.ceil(len(train_set) / dataset_opt['batch_size_per_gpu'])
            total_epochs = math.ceil(opt['train']['total_iter'] / num_iter_per_epoch)
        elif phase == 'val':
            dataset_opt['is_train'] = False
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set, dataset_opt)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized')

    return train_loader, val_loader, total_epochs


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = f"{opt['name']}_train_ckpt"
        if osp.isdir(state_path):
            states = list(os.listdir(state_path))
            if len(states) > 0:
                states = [float(s.split('.state')[0]) for s in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['state_path'] = resume_state_path
    else:
        resume_state_path = opt['state_path']

    if resume_state_path is None:
        resume_state = None
    else:
        devide_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))

    return resume_state


def train_pipeline(root_path):
    opt = parse_options(root_path, is_train=True)
    opt['root_dir'] = root_path

    train_opt = opt['train']
    total_iter = train_opt['total_iter']
    device = get_device()

    # log_dir = opt['path']['log']
    # if not osp.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_file = osp.join(log_dir, f"train_{opt['name']}.log")
    # logger = get_logger(logger_name=opt['name'], log_level=logging.INFO, log_file=log_file)

    if opt['logger']['use_tb_logger']:
        tb_logger_dir = osp.join(opt['path']['tb_logger'], opt['name'])
        if not osp.exists(tb_logger_dir):
            os.makedirs(tb_logger_dir)
        tb_logger = init_tb_logger(tb_logger_dir)

    model = build_model(opt)

    start_epoch = 0
    current_iter = 0

    train_loader, val_loader, total_epochs = create_train_val_dataloader(opt)

    start_time = time.time()
    for epoch in range(start_epoch, total_epochs+1):

        for sample in train_loader:
            current_iter += 1
            if current_iter>total_iter:
                break

            model.update_learning_rate(current_iter, train_opt['warmup_iter'])
            model.feed_data(sample)
            model.optimize_parameters(current_iter)

            if model.cri_pix:
                tb_logger.add_scalar(f"losses/{train_opt['pixel_opt']['type']}",
                                    model.loss_dict['l_pix'], current_iter)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                model.save(current_iter)

            if opt.get('val') is not None and (current_iter % opt['logger']['print_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger)


if __name__=='__main__':
    train_pipeline('./VSR')
