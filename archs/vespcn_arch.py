# Modify from
# https://github.com/JuheonYi/VESPCN-PyTorch/blob/master/model/motioncompensator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.arch_util import flow_warp, make_layer
from archs.espcn_arch import ESPCN
from utils.registry import ARCH_REGISTRY


class SimpleBlock(nn.Module):
    def __init__(self):
        super(SimpleBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=24,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class MotionCompensator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=24):
        super(MotionCompensator, self).__init__()
        self.coarse_flow = nn.Sequential(
            nn.Conv2d(2*num_in_ch, num_feat, 5, 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 5, 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, 32, 3, 1), nn.Tanh(), nn.PixelShuffle(4))

        self.fine_flow = nn.Sequential(
            nn.Conv2d((3*num_in_ch)+2, num_feat, 5, 2, 2), nn.ReLU(inplace=True),
            make_layer(SimpleBlock, 3),
            nn.Conv2d(num_feat, 8, 3, 1, 1), nn.Tanh(), nn.PixelShuffle(2))

    def forward(self, x1, x2):
        coarse_in = torch.cat((x1, x2), dim=1)
        coarse_out = self.coarse_flow(coarse_in)
        # coarse_out[:, 0] /= x1.shape[3]
        # coarse_out[:, 1] /= x2.shape[2]
        x2_compensated_coarse = flow_warp(x2, torch.permute(coarse_out, (0,2,3,1)))

        fine_in = torch.cat((x1, x2, x2_compensated_coarse, coarse_out), dim=1)
        fine_out = self.fine_flow(fine_in)
        # fine_out[:, 0] /= x1.shape[3]
        # fine_out[:, 1] /= x2.shape[2]
        flow = (coarse_out + fine_out)

        x2_compensated = flow_warp(x2, torch.permute(flow, (0,2,3,1)))

        return x2_compensated


@ARCH_REGISTRY.register()
class VESPCN(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_ch, upscale=4):
        super(VESPCN, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_ch = num_ch
        self.upscale = upscale
        self.motion_compensator = MotionCompensator()
        self.espcn = ESPCN(self.num_in_ch, self.num_out_ch, self.num_ch, self.upscale)

    def forward(self, frames):
        b, t, c, h, w = frames.size()
        x1 = frames[:, 0, :, :, :]
        x2 = frames[:, 1, :, :, :]
        x3 = frames[:, 2, :, :, :]

        x1_compensated = self.motion_compensated(x2, x1)
        x3_compensated = self.motion_compensated(x2, x3)

        lr = torch.cat((x1_compensated, x2, x3_compensated), dim=0)

        out = self.espcn.forward(lr)
        out = out.unsqueeze(0).reshape(b, t, c, h*4, w*4)
        return out[: 1, :, :, :]
