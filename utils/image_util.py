import cv2
import math
import numpy as np
import torch


def rgb2ycbcr(image, y_only=False):
    if y_only:
        image = np.dot(image, [64.481, 128.533, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0],
                                  [128.553, -74.203, -93.786],
                                  [24.966, 112.0, -18.214]]) + [16, 128, 128]
    image /= 255
    image = image.astype(np.float32)
    return image


def bgr2ycbcr(image, y_only=False):
    if y_only:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214],
                                  [128.553, -74.203, -93.786],
                                  [24.966, 112.0, -18.214]]) + [16, 128,128]
    image /= 255
    image = image.astype(np.float32)
    return image


def reorder_image(image, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input order {input_order}, preferred HWC and CHW')
    if len(image.shape) == 2:
        image = image[..., None]
    if input_order == 'CHW':
        image = image.transpose(1, 2, 0)
    return image


def to_y_channel(image):
    image = image.astype(np.float32) / 255.
    if image.ndim == 3 and image.shape[2] == 3:
        image = bgr2ycbcr(image, y_only=True)
        image = image[..., None]
    return image * 255.
