import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from conv import BasicBlock, ConvBlock


class ASPPNeck(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pre_conv = BasicBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=in_channels,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        # )
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
        self.post_conv = ConvBlock(in_channels * 4, in_channels, kernel_size=1, stride=1)

    def _Forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        branch1 = F.conv2d(x, self.weight, stride=1, bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1, bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1, bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1, bias=None, padding=18, dilation=18)
        x = self.post_conv(torch.cat((branch1, branch6, branch12, branch18), dim=1))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            out = cp.checkpoint(self._Forward, x, use_reentrant=False)
        else:
            out = self._Forward(x)
        return out
