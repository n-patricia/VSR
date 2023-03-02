# https://github.com/ckkelvinchan/offset-fidelity-loss/blob/main/offset_fidelity_loss.py
# Understanding Deformable Alignment in Video Super-Resolution, AAAI, 2021

import torch
import torch.nn as nn

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class OffsetFidelityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, threshold=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.threshold = threshold

    def forward(self, offset, flow):
        b, c, h, w = offset.size()
        offset = offset.view(-1, 2, h, w) # separate offset in batch dimension

        # flip and repeat the optical flow
        flow = flow.flip(1).repeat(1, c//2, 1, 1).view(-1, 2, h, w)

        # compute loss
        abs_diff = torch.abs(offset - flow)
        mask = (abs_diff > self.threshold).type_as(abs_diff)
        loss = torch.sum(torch.mean(mask * abs_diff, dim=(1, 2, 3)))

        return self.loss_weight * loss
