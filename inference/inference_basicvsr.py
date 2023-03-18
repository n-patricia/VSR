import argparse
import cv2
import glob
import numpy as np
import os
import os.path as osp
import shutil

import torch

from archs.basicvsr_arch import BasicVSR
from utils import tensor2img


def inference(images, image_names, model, save_path):
    with torch.no_grad():
        outputs = model(images)
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, iname in zip(outputs, image_names):
        output = tensor2img(output)
        cv2.imwrite(osp.join(save_path, f'{iname}_BasicVSR.png'), output)


def get_images(path):
    images = [cv2.imread(v).astype(np.float32) / 255. for v in paths]
    images = img2tensor(images, bgr2rgb=True, float32=True)
    images = torch.stack(images, dim=0)
    image_names = [osp.splitext(osp.basename(p))[0] for p in path]
    return images, image_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='experiments/pretrained/BasicVSR_REDSx4_latest.pth')
    parser.add_argument('--input_path', type=str,
                        default='../datasets/REDS/sharp_bicubic/000',
                        help='input test image folder')
    parser.add_argument('--save_path', type=str, default='./results/BasicVSR')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = BasicVSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = osp.splitext(osp.split(args.input_path)[-1])[0]
        input_path = osp.join('./BasicVSR_tmp', video_name)
        os.makedirs(osp.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1'\
                  '-vsync 0 {input_path} /frame%08d.png')

    image_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_images = len(images_list)
    if len(images_list) <= args.interval:
        images, image_names = get_images(image_list[idx:idx+interval])
        images = images.unsqueeze(0).to(device)
        inference(images, image_names, model, args.save_path)

    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__=='__main__':
    main()
