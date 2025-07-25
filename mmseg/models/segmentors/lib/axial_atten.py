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
    def __init__(self,in_channel,out_channel):
        super(AA_kernel,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self._init_layers()
        
    def _init_layers(self, device=None):
        self.conv0 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, bias=False)
        self.conv1 = nn.ModuleList()
        self.conv1.append(nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, bias=False))
        self.conv1.append(nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, bias=False))
        self.bn0 = nn.ModuleList()
        self.bn0.append(nn.BatchNorm2d(self.out_channel))
        self.bn0.append(nn.BatchNorm2d(self.out_channel))
        
        self.conv2 = nn.ModuleList()
        self.conv2.append(nn.Conv2d(self.out_channel, 1, kernel_size=1, bias=False))
        self.conv2.append(nn.Conv2d(self.out_channel, 1, kernel_size=1, bias=False))
        
        self.conv3 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        
        self.relu = nn.ReLU(True)
        
        # Chuyển tất cả các layer sang device cần thiết
        if device is not None:
            self.conv0.to(device)
            for layer in self.conv1:
                layer.to(device)
            for layer in self.bn0:
                layer.to(device)
            for layer in self.conv2:
                layer.to(device)
            self.conv3.to(device)
            self.bn1.to(device)
            self.relu.to(device)
            
    def forward(self,x):
        # Kiểm tra nếu số kênh của input thay đổi, cần tái tạo các layer
        if x.size(1) != self.in_channel:
            self.in_channel = x.size(1)
            # Nếu in_channel thay đổi, thì out_channel cũng cần cập nhật để tránh mismatch
            self.out_channel = x.size(1)
            self._init_layers(device=x.device)
        
        x_size = x.size()
        
        # Qua conv0
        x_conv0 = self.conv0(x)
        
        # Horizontal branch
        h_branch = x_conv0.permute(0, 1, 3, 2)  # Swap H and W
        h_branch = self.conv1[0](h_branch)
        h_branch = self.bn0[0](h_branch)
        h_branch = self.relu(h_branch)
        h_branch = self.conv2[0](h_branch)
        h_attn = h_branch.permute(0, 1, 3, 2)  # Swap back H and W
        
        # Vertical branch
        v_branch = self.conv1[1](x_conv0)
        v_branch = self.bn0[1](v_branch)
        v_branch = self.relu(v_branch)
        v_branch = self.conv2[1](v_branch)
        v_attn = v_branch
        
        # Combine attention maps
        spatial_attn = h_attn * v_attn
        spatial_attn = torch.sigmoid(spatial_attn)
        
        # Apply attention to input
        conv3_in = x * spatial_attn
        conv3_out = self.conv3(conv3_in)
        conv3_out = self.bn1(conv3_out)
        conv3_out = self.relu(conv3_out)
        
        return conv3_out
        

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        self._init_layers()
        
    def _init_layers(self, device=None):
        self.conv_query = nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(self.in_channels, self.in_channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Chuyển tất cả các layer sang device cần thiết
        if device is not None:
            self.conv_query.to(device)
            self.conv_key.to(device)
            self.conv_value.to(device)
        
    def forward(self, x):
        # Kiểm tra nếu số kênh của input thay đổi, cần tái tạo các layer
        if x.size(1) != self.in_channels:
            self.in_channels = x.size(1)
            self._init_layers(device=x.device)
        
        batch_size, C, height, width = x.size()
        # Flatten để thành shape [B, C, N]
        proj_query = self.conv_query(x).view(batch_size, -1, width*height).permute(0, 2, 1)  # B x N x C
        proj_key = self.conv_key(x).view(batch_size, -1, width*height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N
        proj_value = self.conv_value(x).view(batch_size, -1, width*height)  # B x C x N
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        out = self.gamma * out + x
        return out