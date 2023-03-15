import os.path as osp
from collections import Counter

import torch

from data.data_util import tensor2img
from models.sr_model import SRModel
from utils import get_logger
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
            torch.cuda.:e empty_cache()

            for k, v in enumerate(self.opt['val']['metrics'].values()):
                result = self._calculate_metrics(metric_data, v)
                self.metric_results[folder][int(frame_idx), k] += result

            # tb_logger.add_scalar(f'metrics/{folder}',
            #                      self.metric_results[folder].mean(),
            #                      current_iter)

        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        metric_results_avg = {folder: torch.mean(tensor, dim=0).cpu() 
                              for (folder, tensor) in self.metric_results.items()}
        
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric[folder][idx].item()

        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)

        log_str = f"Validation {dataset_name}\n"
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f"\t # {metric}: {value:.4f}"
            for folder, tensor in metric_results_avg.items():
                log_str += f"\t # {folder}: {tensor[metric_idx].item():.4f}"

        logger = get_logger()
        logger.info(log_str)

        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f"metrics/{metric}/{folder}", tensor[metric_idx].item(), 
                                         current_iter)