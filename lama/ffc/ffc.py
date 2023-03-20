# References
    # https://medium.com/mlearning-ai/fast-fourier-convolution-a-detailed-view-a5149aae36c4
    # https://github.com/advimman/lama/blob/main/saicinpainting/training/modules/ffc.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralTransformer(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super().__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch //2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_ch // 2, out_ch // 2, groups, **fu_kwargs)
        # if self.enable_lfu:
        self.lfu = FourierUnit(out_ch // 2, out_ch // 2, groups)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        groups=1,
    ):
        super().__init__()

        self.groups = groups

        self.conv = nn.Conv2d(
            in_ch * 2,
            out_ch * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = torch.randn((4, 256, 64, 128))
        b, c, h, w = x.shape

        # (b, c, h, w // 2 + 1)
        ffted = torch.fft.rfftn(x, dim=(2, 3), norm="ortho")
        y_r = ffted.real
        y_i = ffted.imag
        # (b, c * 2, h, w // 2 + 1)
        x = torch.cat([y_r, y_i], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # (b, c, h, w // 2 + 1)
        y_r, y_i = torch.split(x, split_size_or_sections=c, dim=1)
        # (b, c, h, w)
        x = torch.fft.irfftn(torch.complex(y_r, y_i), s=(64, 128), dim=(2, 3), norm="ortho").shape
        return x


class FFC(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        alpha_in,
        alpha_out,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        enable_lfu=True,
        padding_type='reflect',
        gated=False,
        **spectral_kwargs
    ):
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.global_in_num = in_ch_g

        in_ch_g = int(in_ch * alpha_in)
        in_ch_l = in_ch - in_ch_g
        out_ch_g = int(out_ch * alpha_out)
        out_ch_l = out_ch - out_ch_g
        #groups_g = 1 if groups == 1 else int(groups * alpha_out)
        #groups_l = 1 if groups == 1 else groups - groups_g

        module = nn.Identity if in_ch_l == 0 or out_ch_l == 0 else nn.Conv2d
        self.convl2l = module(in_ch_l, out_ch_l, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_ch_l == 0 or out_ch_g == 0 else nn.Conv2d
        self.convl2g = module(in_ch_l, out_ch_g, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_ch_g == 0 or out_ch_l == 0 else nn.Conv2d
        self.convg2l = module(in_ch_g, out_ch_l, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_ch_g == 0 or out_ch_g == 0 else SpectralTransformer
        self.convg2g = module(
            in_ch_g, out_ch_g, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_ch_g == 0 or out_ch_l == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_ch, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.alpha_out != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.alpha_out != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg
