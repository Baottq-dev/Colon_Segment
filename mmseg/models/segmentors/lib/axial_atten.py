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
        
    def _init_layers(self):
        """Khởi tạo hoặc tái tạo các lớp theo kích thước kênh hiện tại"""
        self.conv0 = Conv(self.in_channel, self.out_channel, kSize=1,stride=1,padding=0)
        self.conv1 = Conv(self.out_channel, self.out_channel, kSize=(3, 3),stride = 1, padding=1)
        self.Hattn = self_attn(self.out_channel, mode='h')
        self.Wattn = self_attn(self.out_channel, mode='w')

    def forward(self, x):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if x.size(1) != self.in_channel:
            print(f"AA_kernel: Auto-adjusting for channel size change: {self.in_channel} -> {x.size(1)}")
            self.in_channel = x.size(1)
            # Điều chỉnh kênh ra nếu trước đây nó bằng kênh vào
            if self.out_channel == self.in_channel:
                self.out_channel = x.size(1)
            self._init_layers()
            
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx