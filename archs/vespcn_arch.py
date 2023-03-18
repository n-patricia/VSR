# Modified from
# https://github.com/HighVoltageRocknRoll/sr/blob/master/models/model_vespcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.arch_util import flow_warp, make_layer
from utils.registry import ARCH_REGISTRY


class SimpleBlock(nn.Module):
    def __init__(self):
        super(SimpleBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=24,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class CoarseFlow(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=24, upscale=4):
        super(CoarseFlow, self).__init__()
        self.coarse_flow = nn.Sequential(
            nn.Conv2d(2*num_in_ch, num_feat, 5, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 5, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, 32, 3, 1),
            nn.Tanh(), nn.PixelShuffle(upscale))

    def forward(self, x):
        out = self.coarse_flow(x)
        return out


class FineFlow(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=24, upscale=2):
        self.fine_flow = nn.Sequential(
            nn.Conv2d((3*num_in_ch)+2, num_feat, 5, 2, 2), nn.ReLU(inplace=True),
            make_layer(SimpleBlock, 3, num_feat=num_feat),
            nn.Conv2d(num_feat, 8, 3, 1, 1),
            nn.Tanh(), nn.PixelShuffle(upscale))

    def forward(self, x):
        out = self.fine_flow(x)
        return out


@ARCH_REGISTRY.register()
class VESPCN(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_ch=64, num_feat=24, upscale=4):
        super(VESPCN, self).__init__()
        self.coarse_block = CoarseFlow(num_in_ch, num_feat)
        self.fine_block = FineFlow(num_in_ch, num_feat)
        num_out_ch = int(num_out_ch*(upscale**2))
        self.feature_maps = nn.Sequential(
            nn.Conv2d(num_in_ch, num_ch, 5, 1, 2), nn.Tanh(),
            nn.Conv2d(num_ch, num_ch, 3, 1, 1), nn.Tanh(),
            nn.Conv2d(num_ch, num_ch, 3, 1, 1), nn.Tanh(),
            nn.Conv2d(num_ch, num_ch//2, 3, 1, 1), nn.Tanh(),)
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(num_ch//2, num_out_ch, 3, 1, 1),
            nn.PixelShuffle(upscale))

    def forward(self, x):
        b, t, c, h, w = x.size()
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        x3 = x[:, 2, :, :, :]

        x_lq = torch.cat((
            torch.stack((x2, x1), dim=1),
            torch.stack((x2, x3), dim=1)), dim=0)
        neighbors = torch.cat((x1, x3), dim=0)

        coarse_flow = self.coarse_block(x_lq.view(b*2,c*2,h,w))
        warped_frames = flow_warp(neighbors, torch.permute(coarse_flow,(0,2,3,1)))

        x_compensated = torch.cat((x_lq.view(b*2,c*2,h,w),coarse_flow), dim=1)
        x_compensated = torch.cat((x_compensated,warped_frames), dim=1)
        fine_flow = self.fine_block(x_compensated)
        flow = coarse_flow + fine_flow

        warped_frames = flow_warp(neighbors, torch.permute(flow,(0,2,3,1)))
        warped_frames = warped_frames.view(b,t-1,c,h,w)
        x_sr = torch.stack((warped_frames[:,0,:,:,:], x2, warped_frames[:,1,:,:,:]), dim=1)

        out = self.feature_maps(x_sr.view(-1,c,h,w))
        out = self.sub_pixel(out)
        out = out.view(b,t,-1,h*4,w*4)
        return out[:,1,:,:,:]
