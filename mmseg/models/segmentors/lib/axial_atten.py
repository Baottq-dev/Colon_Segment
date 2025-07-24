# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layer import Conv
from .self_attention import self_attn
import math


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self._init_layers()
        
    def _init_layers(self, device=None):
        """Khởi tạo hoặc tái tạo các lớp theo kích thước kênh hiện tại"""
        self.conv0 = Conv(self.in_channel, self.out_channel, kSize=1,stride=1,padding=0)
        self.conv1 = Conv(self.out_channel, self.out_channel, kSize=(3, 3),stride = 1, padding=1)
        self.Hattn = self_attn(self.out_channel, mode='h')
        self.Wattn = self_attn(self.out_channel, mode='w')
        
        # Chuyển các modules đến device chỉ định
        if device is not None:
            self.conv0.to(device)
            self.conv1.to(device)
            self.Hattn.to(device)
            self.Wattn.to(device)

    def forward(self, x):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if x.size(1) != self.in_channel:
            print(f"AA_kernel: Auto-adjusting for channel size change: {self.in_channel} -> {x.size(1)}")
            self.in_channel = x.size(1)
            # Điều chỉnh kênh ra nếu trước đây nó bằng kênh vào
            if self.out_channel == self.in_channel:
                self.out_channel = x.size(1)
            # Tạo lại các lớp trên cùng device với input
            self._init_layers(device=x.device)
            
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx
        
class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode
        self.in_channels = in_channels
        self._init_layers()
        
    def _init_layers(self, device=None):
        """Khởi tạo hoặc tái tạo các lớp theo kích thước kênh hiện tại"""
        self.query_conv = Conv(self.in_channels, self.in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(self.in_channels, self.in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(self.in_channels, self.in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        
        # Chuyển các modules đến device chỉ định
        if device is not None:
            self.query_conv.to(device)
            self.key_conv.to(device)
            self.value_conv.to(device)
            self.gamma.to(device)
            
    def forward(self, x):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if x.size(1) != self.in_channels:
            print(f"self_attn: Auto-adjusting for channel size change: {self.in_channels} -> {x.size(1)}")
            self.in_channels = x.size(1)
            # Tạo lại các lớp trên cùng device với input
            self._init_layers(device=x.device)
            
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out