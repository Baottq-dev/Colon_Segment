# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:12:46 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        
        # Lưu lại tham số để có thể tái tạo module khi cần thiết
        self.nIn = nIn
        self.nOut = nOut
        self.kSize = kSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bn_acti = bn_acti
        self.bias = bias
        
        self._init_layers()
            
    def _init_layers(self):
        """Khởi tạo hoặc tái tạo các lớp với tham số hiện tại"""
        self.conv = nn.Conv2d(self.nIn, self.nOut, kernel_size=self.kSize,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups, bias=self.bias)
        
        if self.bn_acti:
            self.bn_relu = BNPReLU(self.nOut)
            
    def forward(self, input):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if input.size(1) != self.nIn:
            print(f"Conv: Auto-adjusting for channel size change: {self.nIn} -> {input.size(1)}")
            # Kiểm tra nếu nOut cũng cần điều chỉnh theo tỷ lệ
            if self.nOut == self.nIn:
                self.nOut = input.size(1)
            elif self.groups == self.nIn:  # Depthwise convolution
                self.nOut = self.nOut * (input.size(1) // self.nIn)
                self.groups = input.size(1)
                
            self.nIn = input.size(1)
            self._init_layers()
            
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output
    
    
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.nIn = nIn
        self._init_layers()
    
    def _init_layers(self):
        """Khởi tạo hoặc tái tạo các lớp với kích thước kênh hiện tại"""
        self.bn = nn.BatchNorm2d(self.nIn, eps=1e-3)
        self.acti = nn.PReLU(self.nIn)

    def forward(self, input):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if input.size(1) != self.nIn:
            print(f"BNPReLU: Auto-adjusting for channel size change: {self.nIn} -> {input.size(1)}")
            self.nIn = input.size(1)
            self._init_layers()
            
        output = self.bn(input)
        output = self.acti(output)
        
        return output