from collections import OrderedDict
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from archs import build_network
from data.data_util import tensor2img
from losses import build_loss
from models.base_model import BaseModel
import utils
from utils import get_logger
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.net = build_network(opt['network']).to(self.device)

        # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network', None)
        # if load_path is not None:
        #     self.load_network(self.net, load_path)

        if self.is_train:
            self._init_training()

    def _init_training(self):
        self.net.train()

        train_opt = self.opt['train']
        # if train_opt.get('pixel_opt')['type'] == 'L1Loss':
        #     self.cri_pix = nn.L1Loss()
        # elif train_opt.get('pixel_opt')['type'] == 'MSELoss':
        #     self.cri_pix = nn.MSELoss()
        # elif train_opt.get('pixel_opt')['type'] == 'SmoothL1Loss':
        #     self.cri_pix = nn.SmoothL1Loss(reduction=train_opt.get('pixel_opt')['reduction'])
        # else:
        #     self.cri_pix = None
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self._get_optimizer(optim_type, optim_params, train_opt['optim']['lr'])

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.output = self.net(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.loss_dict = loss_dict
        self.optimizer.step()

    def save(self, current_iter):
        self.save_network(self.net, current_iter)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output = self.net(self.lq)
        self.net.train()

    def validation(self, dataloader, current_iter, tb_logger):
        dataset_name = dataloader.dataset.opt['name']
        if not hasattr(self, 'metric_results'):
            self.metric_results = {metric:0 for metric in self.opt['val']['metrics'].keys()}

        metric_data = dict()
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self._get_current_visuals()
            sr_image = tensor2img([visuals['result']])
            metric_data['image1'] = sr_image
            if 'gt' in visuals:
                gt_image = tensor2img([visuals['gt']])
                metric_data['image2'] = gt_image
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            for k, v in self.opt['val']['metrics'].items():
                self.metric_results[k] += self._calculate_metrics(metric_data, v)

        for metric in self.metric_results.keys():
            self.metric_results[metric] /= (idx+1)
            tb_logger.add_scalar(f'metrics/{metric}',
                                 self.metric_results[metric], current_iter)

    def _get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def _calculate_metrics(self, data, opt):
        metric_type = opt.get('type')
        metric = getattr(utils, metric_type)
        return metric(data['image1'], data['image2'], **opt)

