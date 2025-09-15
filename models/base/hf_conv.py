import torch
import torch.nn as nn


class HFConv2d(nn.Module):
    def __init__(self, inc, outc, learnable=False):
        super(HFConv2d, self).__init__()
        self.pre_filter = torch.nn.Conv2d(inc, outc, 3, padding=1, groups=inc, bias=False)
        filter_kernel = torch.tensor(
            [
                [[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
            ]
        )
        filter_kernel = filter_kernel.repeat(inc, 1, 1, 1)
        self.pre_filter.weight.data = filter_kernel
        self.pre_filter.weight.requires_grad = learnable
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pre_filter(x)
        x = self.out_conv(x)
        
        return x