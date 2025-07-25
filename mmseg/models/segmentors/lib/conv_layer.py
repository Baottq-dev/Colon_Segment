# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:12:46 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, padding=0, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
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
        
    def _init_layers(self, device=None):
        self.conv = nn.Conv2d(self.nIn, self.nOut, self.kSize, 
                             stride=self.stride, padding=self.padding, 
                             bias=self.bias, groups=self.groups, 
                             dilation=self.dilation)
        if self.bn_acti:
            self.bn_relu = BNPReLU(self.nOut)
        
        # Chuyển các layers sang device chỉ định nếu cần
        if device is not None:
            self.conv.to(device)
            if self.bn_acti:
                self.bn_relu.to(device)
                
    def forward(self, input):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if input.size(1) != self.nIn:
            self.nIn = input.size(1)
            
            # Nếu số kênh ra bằng số kênh vào hoặc nếu groups=nIn (depthwise conv)
            # thì số kênh ra và groups cũng phải được thay đổi
            if self.nOut == self.nIn or self.groups == self.nIn:
                self.nOut = self.nIn
            if self.groups == self.nIn:
                self.groups = self.nIn
                
            self._init_layers(device=input.device)
            
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_relu(output)
            
        return output
    
    
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.nIn = nIn
        self._init_layers()
        
    def _init_layers(self, device=None):
        self.bn = nn.BatchNorm2d(self.nIn, eps=1e-3)
        self.acti = nn.PReLU(self.nIn)
        
        # Chuyển các layers sang device chỉ định nếu cần
        if device is not None:
            self.bn.to(device)
            self.acti.to(device)
        
    def forward(self, input):
        # Kiểm tra xem kích thước kênh đầu vào có thay đổi hay không
        if input.size(1) != self.nIn:
            self.nIn = input.size(1)
            self._init_layers(device=input.device)
            
        output = self.bn(input)
        output = self.acti(output)
        
        return output