##################################################################
# modify from
# https://github.com/Lornatang/ESPCN-PyTorch/blob/master/model.py
################################################################## 

import math
import torch
import torch.nn as nn

from utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ESPCN(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_ch, upscale=4):
        super(ESPCN, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_ch = num_ch
        self.upscale = upscale
        hidden_channels = self.num_ch // 2
        self.num_out_ch = int(self.num_out_ch * (self.upscale ** 2))

        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels=self.num_in_ch, out_channels=self.num_ch,
                      kernel_size=5, stride=1, padding=2), nn.Tanh(),
            nn.Conv2d(in_channels=self.num_ch, out_channels=hidden_channels,
                      kernel_size=3, stride=1, padding=1), nn.Tanh(),)

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=self.num_out_ch,
                      kernel_size=3, stride=1,
                      padding=1), nn.PixelShuffle(self.upscale),)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels==32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data, 0.0,
                                    math.sqrt(2/(module.out_channels*module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)


    def forward(self, x):
        x = self.feature_maps(x)
        x = self.sub_pixel(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
