# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/paired_image_dataset.py

import os
import os.path as osp
import cv2
import numpy as np

import torch.utils.data as data

from data.data_util import augment, paired_random_crop
from utils import img2tensor
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.gt_dir, self.lq_dir = self.opt['gt_dir'], self.opt['lq_dir']
        self.scale = self.opt['scale']
        self.is_train = self.opt['is_train']
        self.gt_size = 0
        if self.is_train:
            self.gt_size = self.opt['gt_size']

        # self.y_channel = self.opt['y_channel']
        self.gt_files = [osp.join(self.gt_dir, x) for x in sorted(os.listdir(self.gt_dir))]
        self.lq_files = [osp.join(self.lq_dir, x) for x in sorted(os.listdir(self.lq_dir))]

    def __getitem__(self, index):
        scale = self.opt['scale']
        image_gt = cv2.imread(self.gt_files[index]).astype(np.float32) / 255.
        image_lq = cv2.imread(self.lq_files[index]).astype(np.float32) / 255.

        if self.is_train:
            image_gt, image_lq = paired_random_crop(image_gt, image_lq, self.gt_size, self.scale)
            image_gt, image_lq = augment(image_gt), augment(image_lq)

        # if self.y_channel:
        #     image_gt = bgr2ycbcr(image_gt, y_only=True)[..., None]
        #     image_lq = bgr2ycbcr(image_lq, y_only=True)[..., None]

        image_gt, image_lq = img2tensor(image_gt), img2tensor(image_lq)

        return {'lq': image_lq, 'gt': image_gt}

    def __len__(self):
        return len(self.gt_files)
