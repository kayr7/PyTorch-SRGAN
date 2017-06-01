from __future__ import division

import math

import torch
import torch.nn as nn

# this is one block for a resnet


class Residual(nn.Module):
    def __init__(self, n_channels=64):
        super(Residual, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=self.n_channels,
                               out_channels=self.n_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.n_channels,
                               out_channels=self.n_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.n_channels)

    def forward(self, x):
        input = x
        output = torch.add(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))),
                           input)
        return output


class SubPixelConv(nn.Module):
    def __init__(self, n_channels=64, upsample=2):
        super(SubPixelConv, self).__init__()
        self.n_channels = n_channels
        self.upsample = upsample
        self.out_channels = self.upsample * self.upsample * self.n_channels

        self.conv = nn.Conv2d(in_channels=self.n_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.upsample_net = nn.PixelShuffle(self.upsample)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        output = self.relu(self.upsample_net(self.conv(x)))
        return output


class SRResNet(nn.Module):
    def __init__(self, n_channels=64, n_blocks=15):
        super(SRResNet, self).__init__()
        self.n_channels = n_channels
        self.inConv = nn.Conv2d(in_channels=3,  # RGB
                                out_channels=self.n_channels,
                                kernel_size=3,  # in paper it is 9, somehow other implementations always used 3
                                stride=1,
                                padding=1,
                                bias=True)
        self.inRelu = nn.ReLU(inplace=True)

        self.resBlocks = self.make_block_layers(n_blocks, Residual)

        self.glueConv = nn.Conv2d(in_channels=self.n_channels,
                                  out_channels=self.n_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=True)
        self.glueBN = nn.BatchNorm2d(self.n_channels)

        self.upscaleBlock = self.make_block_layers(2, SubPixelConv)

        self.outConv = nn.Conv2d(in_channels=n_channels,
                                 out_channels=3,  # RGB
                                 kernel_size=3,  # paper has 9
                                 padding=1,
                                 bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        first_step = self.inRelu(self.inConv(x))
        residual = first_step
        output = torch.add(self.glueBN(self.glueConv(self.resBlocks(first_step))),
                           residual)
        output = self.outConv(self.upscaleBlock(output))
        return output

    def make_block_layers(self, n_blocks, block_fn):
        layers = [block_fn() for x in range(n_blocks)]
        return nn.Sequential(*layers)
