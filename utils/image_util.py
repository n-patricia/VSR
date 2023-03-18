import cv2
import math
import numpy as np
import torch
from torchvision.utils import make_grid



def img2tensor(images, bgr2rgb=True, float32=True):
    def _totensor(image, bgr2rgb, float32):
        if image.shape[2] == 3 and bgr2rgb:
            if image.dtype == 'float64':
                image = image.astype('float32')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        if float32:
            image = image.float()
        return image

    if isinstance(images, list):
        return [_totensor(image, bgr2rgb, float32) for image in images]
    else:
        return _totensor(images, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0,1)):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and
                                        all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            image_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))),
                                normalize=False).numpy()
            image_np = image_np.transpose(1, 2, 0)
            if rgb2bgr:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            image_np = _tensor.numpy()
            image_np = image_np.transpose(1, 2, 0)
            if image_np.shape[2] == 1:
                image_np = np.squeeze(image_np, axis=2)
            else:
                if rgb2bgr:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            image_np = _tensor.numpy()
        else:
            raise TypeError(f'Only 4D, 3D, or 2D tensor. But received {n_dim} dim')

        if out_type == np.uint8:
            image_np = (image_np * 255.0).round()
        image_np = image_np.astype(out_type)
        result.append(image_np)

    if len(result) == 1:
        result = result[0]

    return result


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
