import os
import os.path as osp
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import get_logger


class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
        self.is_train = opt['is_train']
        self.scheduler = None
        self.optimizer = None

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def _get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = optim.Adam(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = optim.SGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = optim.RMSprop(params, lr, **kwargs)
        return optimizer

    def _get_current_visuals(self):
        pass

    def setup_schedulers(self):
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, **train_opt['scheduler'])
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, **train_opt['scheduler'])
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet')

    def _set_lr(self, lr_groups_l):
        for optimizer, lr_groups in zip(self.optimizer, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        init_lr_groups_l = []
        for optimizer in self.optimizer:
            init_lr_groups_l.append([p['initial_lr'] for p in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        if current_iter > 1:
            self.scheduler.step()

        if current_iter < warmup_iter:
            init_lr_g_l = self._get_init_lr()
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([i / warmup_iter * current_iter for i in init_lr_g])

            self._set_lr(warm_up_lr_l)

    def get_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def save(self, epoch, current_iter):
        pass

    def validation(self, dataloader, current_iter):
        pass

    def save_network(self, net, current_iter, param_key='params'):
        if not osp.exists(self.opt['path'].get('train_state', None)):
            os.makedirs(self.opt['path'].get('train_state'), None)

        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f"net_{self.opt['model_type']}_x{self.opt['scale']}_iter{current_iter}.pth"
        save_path = osp.join(self.opt['path'].get('train_state', None), save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'): # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

        if param_key is not None:
            load_net = load_net[param_key]

        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

