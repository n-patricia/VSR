import os.path as osp
from collections import Counter

import torch

from data.data_util import tensor2img
from models.video_base_model import VideoBaseModel
from utils import get_logger
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VideoRecurrentModel(VideoBaseModel):
    def __init__(self, opt):
        super(VideoRecurrentModel, self).__init__()
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)

        if flow_lr_mul == 1:
            optim_params = self.net.parameters()
        else:
            normal_params = []
            flow_params = []
            for name, param in self.net.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {'params': normal_params, 'lr': train_opt['optim']['lr']},
                {'params': flow_params, 'lr': train_opt['optim']['lr'] * flow_lr_mul},
            ]

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self._get_optimizer(optim_type, optim_params, **train_opt['optim'])

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            if current_iter == 1:
                # logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters')
                for name, param in self.net.named_parameters():
                    if 'spynet' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                # logger.warning('Train all the parameters')
                self.net.requires_grad_(True)

        super(VideoRecurrentModel, self).optimize_parameters(current_iter)

    def validation(self, dataloader, current_iter, tb_logger):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        if not hasattr(self, 'metric_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(num_frame, len(self.opt['val']['metrics']),
                                                          dtype=torch.float32, device='cuda')

        for _, tensor in self.metric_results.items():
            tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        # num_pad = 0
        for i in range(0, num_folders):
            # idx = min(i, num_folders - 1)
            val_data = dataset[i]
            folder = val_data['folder']

            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.test()
            visuals = self._get_current_visuals()

            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt

            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_image = tensor2img([result])
                    metric_data['image1'] = result_image
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_image = tensor2img([gt])
                        metric_data['image2'] = gt_image

                    for k, v in enumerate(self.opt['val']['metrics'].values()):
                        result = self._calculate_metrics(metric_data, v)
                        self.metric_results[folder][idx, k] += result

            # tb_logger.add_scalar(f'metrics/{folder}',
            #                      self.metric_results[folder].mean(), current_iter)
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        

    def test(self):
        n = self.lq.size(1)
        self.net.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)
        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.net(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n//2, :, :, :]

        self.net.train()
