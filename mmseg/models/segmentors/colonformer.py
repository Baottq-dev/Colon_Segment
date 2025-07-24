import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule

@SEGMENTORS.register_module()
class ColonFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ColonFormer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        # Kiểm tra xem backbone có phải là Swin Transformer không
        is_swin = hasattr(backbone, 'type') and backbone['type'].startswith('swin_')
        
        # Xác định kích thước kênh đầu ra
        if is_swin:
            if backbone['type'] == 'swin_tiny' or backbone['type'] == 'swin_small':
                c1, c2, c3, c4 = 96, 192, 384, 768
            elif backbone['type'] == 'swin_base':
                c1, c2, c3, c4 = 128, 256, 512, 1024
            else:
                c1, c2, c3, c4 = 64, 128, 320, 512  # Default MiT
        else:
            c1, c2, c3, c4 = 64, 128, 320, 512  # Default MiT
        
        self.CFP_1 = CFPModule(c2, d = 8)
        self.CFP_2 = CFPModule(c3, d = 8)
        self.CFP_3 = CFPModule(c4, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(c2,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(c3,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(c4,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(c2,c2)
        self.aa_kernel_2 = AA_kernel(c3,c3)
        self.aa_kernel_3 = AA_kernel(c4,c4)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  64x88x88 hoặc 96x88x88 nếu dùng Swin
        x2 = segout[1]  # 128x44x44 hoặc 192x44x44 nếu dùng Swin
        x3 = segout[2]  # 320x22x22 hoặc 384x22x22 nếu dùng Swin
        x4 = segout[3]  # 512x11x11 hoặc 768x11x11 nếu dùng Swin

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        # Thay vì sử dụng scale_factor cố định, resize theo kích thước của feature map x4
        cfp_out_1 = self.CFP_3(x4)
        print(f"Debug - cfp_out_1 shape: {cfp_out_1.shape}, x4 shape: {x4.shape}")
        
        # Resize decoder_1 về cùng kích thước không gian với cfp_out_1
        decoder_2 = F.interpolate(decoder_1, size=cfp_out_1.shape[2:], mode='bilinear', align_corners=False)
        print(f"Debug - decoder_2 shape: {decoder_2.shape}")
        
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        print(f"Debug - decoder_2_ra shape: {decoder_2_ra.shape}")
        
        # Đảm bảo sử dụng cfp_out_1 mới nhất cho aa_kernel_3
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        print(f"Debug - aa_atten_3 shape after AA_kernel: {aa_atten_3.shape}")
        # Không sử dụng phép cộng trực tiếp nữa, thay vào đó là gán giá trị mới
        aa_atten_3 = aa_atten_3 + cfp_out_1
        print(f"Debug - aa_atten_3 shape after addition: {aa_atten_3.shape}")
        # Lấy số kênh từ cfp_out_1
        _, c4_out, _, _ = cfp_out_1.shape
        aa_atten_3_o = decoder_2_ra.expand(-1, c4_out, -1, -1).mul(aa_atten_3)
        print(f"Debug - aa_atten_3_o shape: {aa_atten_3_o.shape}")
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        cfp_out_2 = self.CFP_2(x3)
        print(f"Debug - cfp_out_2 shape: {cfp_out_2.shape}, x3 shape: {x3.shape}")
        
        # Resize x_3 về cùng kích thước không gian với cfp_out_2
        decoder_3 = F.interpolate(x_3, size=cfp_out_2.shape[2:], mode='bilinear', align_corners=False)
        print(f"Debug - decoder_3 shape: {decoder_3.shape}")
        
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        print(f"Debug - decoder_3_ra shape: {decoder_3_ra.shape}")
        
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        print(f"Debug - aa_atten_2 shape after AA_kernel: {aa_atten_2.shape}")
        # Không sử dụng phép cộng trực tiếp nữa
        aa_atten_2 = aa_atten_2 + cfp_out_2
        print(f"Debug - aa_atten_2 shape after addition: {aa_atten_2.shape}")
        # Lấy số kênh từ cfp_out_2
        _, c3_out, _, _ = cfp_out_2.shape
        aa_atten_2_o = decoder_3_ra.expand(-1, c3_out, -1, -1).mul(aa_atten_2)
        print(f"Debug - aa_atten_2_o shape: {aa_atten_2_o.shape}")
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        cfp_out_3 = self.CFP_1(x2)
        print(f"Debug - cfp_out_3 shape: {cfp_out_3.shape}, x2 shape: {x2.shape}")
        
        # Resize x_2 về cùng kích thước không gian với cfp_out_3
        decoder_4 = F.interpolate(x_2, size=cfp_out_3.shape[2:], mode='bilinear', align_corners=False)
        print(f"Debug - decoder_4 shape: {decoder_4.shape}")
        
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        print(f"Debug - decoder_4_ra shape: {decoder_4_ra.shape}")
        
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        print(f"Debug - aa_atten_1 shape after AA_kernel: {aa_atten_1.shape}")
        # Không sử dụng phép cộng trực tiếp nữa
        aa_atten_1 = aa_atten_1 + cfp_out_3
        print(f"Debug - aa_atten_1 shape after addition: {aa_atten_1.shape}")
        # Lấy số kênh từ cfp_out_3
        _, c2_out, _, _ = cfp_out_3.shape
        aa_atten_1_o = decoder_4_ra.expand(-1, c2_out, -1, -1).mul(aa_atten_1)
        print(f"Debug - aa_atten_1_o shape: {aa_atten_1_o.shape}")
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear') 
        
        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1