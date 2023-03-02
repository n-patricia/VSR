import cv2
import random
import math
import numpy as np
import torch
from torchvision.utils import make_grid


def generate_from_indices(crt_idx, max_frame_num, num_frames,
                          padding='reflection'):
    assert num_frames % 2 == 1, 'num_frames should be an odd number'
    assert padding in ('replicate', 'reflection', 'reflection_circle',
                       'circle'), f'Wrong padding {padding}'
    max_frame_num = max_frame_num - 1
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)

    return indices


def paired_random_crop(image_gts, image_lqs, patch_size, scale):
    if not isinstance(image_gts, list):
        image_gts = [image_gts]
    if not isinstance(image_lqs, list):
        image_lqs = [image_lqs]

    input_type = 'Tensor' if torch.is_tensor(image_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_gt, w_gt = image_gts[0].size()[-2:]
        h_lq, w_lq = image_lqs[0].size()[-2:]
    else:
        h_gt, w_gt = image_gts[0].shape[0], image_gts[0].shape[1]
        h_lq, w_lq = image_lqs[0].shape[0], image_lqs[0].shape[1]
    lq_patch_size = patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches GT ({h_gt}, {w_gt}) is not {scale}x')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size {lq_patch_size}')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop
    if input_type == 'Tensor':
        image_lqs = [x[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for x in image_lqs]
    else:
        image_lqs = [x[top:top + lq_patch_size, left:left + lq_patch_size, ...] for x in image_lqs]

    top_gt, left_gt = int(top * scale), int(left * scale)

    if input_type == 'Tensor':
        image_gts = [x[:, :, top_gt:top_gt + patch_size, left_gt:left_gt + patch_size] for x in image_gts]
    else:
        image_gts = [x[top_gt:top_gt + patch_size, left_gt:left_gt + patch_size, ...] for x in image_gts]

    if len(image_gts) == 1:
        image_gts = image_gts[0]
    if len(image_lqs) == 1:
        image_lqs = image_lqs[0]

    return image_gts, image_lqs


def augment(images, hflip=True, rotation=True, flows=None):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(image):
        if hflip:
            cv2.flip(image, 1, image)
        if vflip:
            cv2.flip(image, 0, image)
        if rot90:
            image = image.transpose(1, 0, 2)
        return image

    def _augment_flow(flow):
        if hflip:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1,0]]
        return flow

    if not isinstance(images, list):
        images = [images]
    images = [_augment(image) for image in images]
    if len(images) == 1:
        images = images[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return images, flows

    return images


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


