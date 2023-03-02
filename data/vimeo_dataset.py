import cv2
import random
import numpy as np

import torch
import torch.utils.data as data

from data.data_util import augment, img2tensor, paired_random_crop
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VimeoDataset(data.Dataset):
    def __init__(self, opt):
        super(VimeoDataset, self).__init__()
        self.opt = opt
        self.gt_dir, self.lq_dir = opt['gt_dir'], opt['lq_dir']

        with open(opt['meta_info_file'], 'r') as f:
            self.keys = [line.rstrip() for line in f]

        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in
                              range(opt['num_frame'])]
        self.random_reverse = opt['random_reverse']

    def __getitem__(self, index):
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')

        image_gt_path = f'{self.gt_dir}/{clip}/{seq}/im4.png'
        image_gt = cv2.imread(image_gt_path).astype(np.float32) / 255.

        image_lqs = []
        for neighbor in self.neighbor_list:
            image_lq_path = f'{self.lq_dir}/{clip}/{seq}/im{neighbor}.png'
            image_lq = cv2.imread(image_lq_path).astype(np.float32) / 255.
            image_lqs.append(image_lq)

        image_gt, image_lqs = paired_random_crop(image_gt, image_lqs, gt_size, scale)
        image_lqs.append(image_gt)

        image_results = augment(image_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        image_results = img2tensor(image_results)

        image_lqs = torch.stack(image_results[0:-1], dim=0)
        image_gt = image_results[-1]

        return {'gt': image_gt, 'lq': image_lqs}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class VimeoRecurrentDataset(data.Dataset):
    def __init__(self, opt):
        super(VimeoRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_dir = opt['gt_dir']
        self.lq_dir = opt['lq_dir']

        with open(opt['meta_info_file'], 'r') as f:
            self.keys = [line.rstrip() for line in f]

        self.neighbor_list = [1, 2, 3, 4, 5, 6, 7]

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')

        image_gts = []
        image_lqs = []
        for neighbor in self.neighbor_list:
            image_gt_path = f'{self.gt_dir}/{clip}/{seq}/im{neighbor}.png'
            image_gt = cv2.imread(image_gt_path).astype(np.float32) / 255.

            image_lq_path = f'{self.lq_dir}/{clip}/{seq}/im{neighbor}.png'
            image_lq = cv2.imread(image_lq_path).astype(np.float32) / 255.

            image_gts.append(image_gt)
            image_lqs.append(image_lq)

        image_gts, image_lqs = paired_random_crop(image_gts, image_lqs, gt_size, scale)
        image_lqs.extend(image_gts)
        image_results = augment(image_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        image_results = img2tensor(image_results)
        image_lqs = torch.stack(image_results[:7], dim=0)
        image_gts = torch.stack(image_results[7:], dim=0)

        if isinstance(image_gts, list):
            image_gts = image_gts[0]
        if isinstance(image_lqs, list):
            image_lqs = image_lqs[0]

        return {'gt': image_gts, 'lq': image_lqs}

    def __len__(self):
        return len(self.keys)
