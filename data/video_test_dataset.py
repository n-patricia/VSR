import cv2
import glob
import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

from data.data_util import generate_from_indices, img2tensor
from utils import get_logger
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        # self.cache_data = opt['cache_data']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': []}
        self.gt_dir, self.lq_dir = opt['gt_dir'], opt['lq_dir']

        logger = get_logger()
        logger.info(f"Generate data info for VideoTestDataset - {opt['name']}")

        self.image_lqs = {}
        self.image_gts = {}
        subfolders = sorted(glob.glob(osp.join(self.gt_dir, '*')))
        for subfolder in subfolders:
            subfolder_name = osp.basename(subfolder)
            path_lqs = self._get_paths(self.lq_dir, subfolder_name)
            path_gts = self._get_paths(self.gt_dir, subfolder_name)
            self.data_info['lq_path'].extend(path_lqs)
            self.data_info['gt_path'].extend(path_gts)

            max_idx = len(path_lqs)
            self.data_info['idx'].extend([f'{i}/{max_idx}' for i in
                                          range(max_idx)])
            self.data_info['folder'].extend([subfolder_name] * max_idx)

            self.image_lqs[subfolder_name] = self._get_images(path_lqs)
            self.image_gts[subfolder_name] = self._get_images(path_gts)

        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        select_idx = generate_from_indices(idx, max_idx, self.opt['num_frame'],
                                           self.opt['padding'])
        image_lqs = [self.image_lqs[folder][i] for i in select_idx]
        image_gt = [self.image_gts[folder][idx]]

        image_lqs = img2tensor(image_lqs)
        image_lqs = torch.stack(image_lqs, dim=0)
        image_gt = img2tensor(image_gt)
        image_gt = torch.stack(image_gt, dim=0)

        return {'gt': image_gt, 'lq': image_lqs,
                'folder': folder, 'idx': self.data_info['idx'][index]}

    def _get_images(self, path_name, folder_name):
        images = []
        for file in sorted(os.listdir(osp.join(path_name, folder_name))):
            img = cv2.imread(osp.join(path_name, folder_name, file)).astype(np.float32)
            img /= 255.
            images.append(img)

        return images

    def _get_paths(self, path_name):
        images = []
        for file in path_name:
            img = cv2.imread(file).astype(np.float32) / 255.
            images.append(img)

        return images

    def __len__(self):
        return len(self.folders)
    


class VideoRecurrentTestDataset(VideoTestDataset):
    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        logger = get_logger()
        logger.info(f"Generate data info for VideoRecurrentTestDataset - {opt['name']}")

        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        image_lqs = self.image_lqs[folder]
        image_lqs = img2tensor(image_lqs)
        image_lqs = torch.stack(image_lqs, dim=0)
        image_gts = self.image_gts[folder]
        image_gts = img2tensor(image_gts)
        image_gts = torch.stack(image_gts, dim=0)

        return {'gt': image_gts, 'lq': image_lqs, 'folder': folder}
    
    def __len__(self):
        return len(self.folders)
