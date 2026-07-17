import math
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d, DeformConv2d  # 不支持转onnx
from torch.nn.modules.utils import _pair
import logging

logger = logging.getLogger('base')


class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class ModulatedDeformConvPack2(ModulatedDeformConv):
    def __init__(self, *args, extra_offset_mask=False, offset_in_channel=32, **kwargs):
        super(ModulatedDeformConvPack2, self).__init__(*args, **kwargs)

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            offset_in_channel,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # print(o1[0,:,o1.size()[2]//2,o1.size()[3]//2])
        # print(o2[0,:,o1.size()[2]//2,o1.size()[3]//2])
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))

        return deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DeformableConv2d(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            offset_groups=1,
            extra_offset_mask=False,
            offset_in_channel=32
    ):
        super(DeformableConv2d, self).__init__()
        # assert in_dim % groups == 0
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.extra_offset_mask = extra_offset_mask

        self.conv_offset_mask = nn.Conv2d(offset_in_channel, 3 * offset_groups * kernel_size * kernel_size,
                                          kernel_size=_pair(kernel_size), stride=_pair(stride),
                                          padding=_pair(padding),
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        oh, ow, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((oh, ow), dim=1)
        mask = torch.sigmoid(mask)
        # if self.with_mask:
        #     oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
        #     offset = torch.cat([oh, ow], dim=1)
        #     mask = mask.sigmoid()
        # else:
        #     offset = self.param_generator(x)
        #     mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x
