import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.ops as ops


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()

    grid_x, grid_y = torch.meshgrid(torch.arange(0, h).type_as(x),
                                    torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False

    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w-1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h-1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode,
                           padding_mode=padding_mode,
                           align_corners=align_corners)

    return output


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwargs):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


# https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
class DeformConv2d(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, kernel_size=3, stride=1,
                 padding=1, deformable_groups=1, bias=False):
        super(DeformConv2d, self).__init__()
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(num_in_ch,
                                     2*deformable_groups*kernel_size[0]*kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=True)

        # nn.init.constant_(self.offset_conv.weight, 0.)
        # nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(num_in_ch,
                                        deformable_groups*kernel_size[0]*kernel_size[1],
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=True)

        # nn.init.constant_(self.modulator_conv.weight, 0.)
        # nn.init.constant_(self.modulator_conv.bias, 0.)
        default_init_weights([self.offset_conv, self.modulator_conv], 0.)

        self.regular_conv = nn.Conv2d(num_in_ch, num_out_ch,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=self.padding, bias=bias)

    def forward(self, x, z):
        offset = self.offset_conv(z)
        modulator = 2.*torch.sigmoid(self.modulator_conv(x))
        out = ops.deform_conv2d(x, offset, weight=self.regular_conv.weight,
                              bias=self.regular_conv.bias,
                              padding=self.padding, mask=modulator,
                              stride=self.stride)

        return out

