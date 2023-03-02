import os.path as osp
from collections import Counter

import torch

from data.data_util import tensor2img
from models.sr_model import SRModel
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VideoBaseModel(SRModel):
    def validation(self, dataloader, current_iter, tb_logger):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']

        if not hasattr(self, 'metric_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(num_frame,
                                                          len(self.opt['val']['metrics']),
                                                          dtype=torch.float32,
                                                          device='cuda')

        for _, tensor in self.metric_results.items():
            tensor.zero_()

        metric_data = dict()
        for idx in range(0, len(dataset)):
            val_data = dataset[idx]
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            folder = val_data['folder']
            frame_idx, _ = val_data['idx'].split('/')

            self.feed_data(val_data)
            self.test()
            visuals = self._get_current_visuals()
            result_image = tensor2img([visuals['result']])
            metric_data['image1'] = result_image
            if 'gt' in visuals:
                gt_image = tensor2img([visuals['gt']])
                metric_data['image2'] = gt_image
                del self.gt

            del self.lq
            del self.output
            torch.cuda_empty_cache()

            for k, v in enumerate(self.opt['val']['metrics'].values()):
                result = self._calculate_metrics(metric_data, v)
                self.metric_results[folder][int(frame_idx), k] += result

            tb_logger.add_scalar(f'metrics/{folder}',
                                 self.metric_results[folder].mean(),
                                 current_iter)
