import cv2
import numpy as np
import torch

from utils import reorder_image, to_y_channel


def calculate_psnr(image1, image2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    if image1.dtype == torch.float:
        image1 = image1.cpu().detach().numpy()
    if image2.dtype == torch.float:
        image2 = image2.cpu().detach().numpy()

    assert image1.shape == image2.shape, f'Image shapes are different {image1.shape} - {image2.shape}'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}, preferred HWC and CHW')
    image1 = reorder_image(image1, input_order=input_order)
    image2 = reorder_image(image2, input_order=input_order)

    if crop_border != 0:
        image1 = image1[crop_border:-crop_border, crop_border:-crop_border, ...]
        image2 = image2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        image1 = to_y_channel(image1)
        image2 = to_y_channel(image2)

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mse = np.mean((image1 - image2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255./mse)


def _ssim(image1, image2):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(image1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(image2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(image1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(image2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(image1 * image2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(image1, image2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    if image1.dtype == torch.float:
        image1 = image1.cpu().detach().numpy()
    if image2.dtype == torch.float:
        image2 = image2.cpu().detach().numpy()

    assert image1.shape == image2.shape, f'Image shapes are different {image1.shape} - {image2.shape}'
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}, preferred HWC and CHW')
    image1 = reorder_image(image1, input_order=input_order)
    image2 = reorder_image(image2, input_order=input_order)

    if crop_border != 0:
        image1 = image1[crop_border:-crop_border, crop_border:-crop_border, ...]
        image2 = image2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        image1 = to_y_channel(image1)
        image2 = to_y_channel(image2)

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    ssims = []
    for i in range(image1.shape[2]):
        ssims.append(_ssim(image1[..., i], image2[..., i]))
    return np.array(ssims).mean()
