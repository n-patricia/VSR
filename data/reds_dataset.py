import numpy as np
import random
import cv2

import torch
import torch.utils.data as data

from data.data_util import augment, img2tensor, paired_random_crop
from utils import get_logger
from utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class REDSDataset(data.Dataset):
    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.gt_dir, self.lq_dir = opt['gt_dir'], opt['lq_dir']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as file:
            for l in file:
                folder, frame_num, _ = l.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; random reverse is {self.random_reverse}')

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')
        center_frame_idx = int(frame_name)

        # determine neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval

        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
            start_frame_idx = (center_frame_idx - self.num_half_frames *
                               interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval

        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1,
                                   interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list')

        image_gt_path = f'{self.gt_dir}/{clip_name}/{frame_name}.png'
        image_gt = cv2.imread(image_gt_path).astype(np.float32) / 255.

        image_lqs = []
        for neighbor in neighbor_list:
            image_lq_path = f'{self.lq_dir}/{clip_name}/{neighbor:08d}.png'
            image_lq = cv2.imread(image_lq_path).astype(np.float32) / 255.
            image_lqs.append(image_lq)

        image_gt, image_lqs = paired_random_crop(image_gt, image_lqs, gt_size, scale)

        image_lqs.append(image_gt)
        image_results = augment(image_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        image_results = img2tensor(image_results)
        image_lqs = torch.stack(image_results[0:-1], dim=0)
        image_gt = image_results[-1]

        return {'gt': image_gt, 'lq': image_lqs, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class REDSRecurrentDataset(data.Dataset):
    def __init__(self, opt):
        super(REDSRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_dir, self.lq_dir = opt['gt_dir'], opt['lq_dir']
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as file:
            for l in file:
                folder, frame_num, _ = l.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in
                                  range(int(frame_num))])

        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        # interval_str = ','.join(str(x) for x in self.interval_list)

    def __getitem__(self, index):
        scale = self.opt['upscale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')

        interval = random.choice(self.interval_list)
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame *
                                             interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        image_lqs = []
        image_gts = []
        for neighbor in neighbor_list:
            image_lq_path = f'{self.lq_dir}/{clip_name}/{neighbor:08d}.png'
            image_lq = cv2.imread(image_lq_path).astype(np.float32) / 255.
            image_lqs.append(image_lq)

            image_gt_path = f'{self.gt_dir}/{clip_name}/{neighbor:08d}.png'
            image_gt = cv2.imread(image_gt_path).astype(np.float32) / 255.
            image_gts.append(image_gt)

        image_gts, image_lqs = paired_random_crop(image_gts, image_lqs,
                                                  gt_size, scale)

        image_lqs.extend(image_gts)
        image_results = augment(image_lqs, self.opt['use_hflip'],
                                self.opt['use_rot'])
        image_results = img2tensor(image_results)
        image_gts = torch.stack(image_results[len(image_lqs)//2:], dim=0)
        image_lqs = torch.stack(image_results[:len(image_lqs)//2], dim=0)

        return {'gt': image_gts, 'lq': image_lqs, 'key': key}

    def __len__(self):
        return len(self.keys)

