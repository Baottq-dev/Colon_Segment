import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import random
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

import albumentations as A
from albumentations.core.composition import Compose, OneOf

from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet

# Mapping backbone tới tên model
BACKBONE_MODEL_MAPPING = {
    # MiT backbones
    'b1': 'ColonFormer-XS',
    'b2': 'ColonFormer-S', 
    'b3': 'ColonFormer-L',
    'b4': 'ColonFormer-XL',
    'b5': 'ColonFormer-XXL',
    # Swin Transformer backbones
    'swin_tiny': 'ColonFormer-Swin-T',
    'swin_small': 'ColonFormer-Swin-S',
    'swin_base': 'ColonFormer-Swin-B'
}

# Mapping backbone tới kích thước kênh đầu ra
BACKBONE_CHANNELS = {
    # MiT backbones
    'b0': [32, 64, 160, 256],
    'b1': [64, 128, 320, 512],
    'b2': [64, 128, 320, 512],
    'b3': [64, 128, 320, 512],
    'b4': [64, 128, 320, 512],
    'b5': [64, 128, 320, 512],
    # Swin Transformer backbones
    'swin_tiny': [96, 192, 384, 768],
    'swin_small': [96, 192, 384, 768],
    'swin_base': [128, 256, 512, 1024]
}

# Cấu hình cho Swin Transformer
SWIN_CONFIGS = {
    'swin_tiny': {
        'embed_dims': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
    },
    'swin_small': {
        'embed_dims': 96,
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
    },
    'swin_base': {
        'embed_dims': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size': 7,
    }
}

def get_model_name(backbone):
    """Lấy tên model từ backbone"""
    return BACKBONE_MODEL_MAPPING.get(backbone, f'ColonFormer-{backbone.upper()}')
    
def is_swin_backbone(backbone):
    """Kiểm tra xem backbone có phải là Swin Transformer không"""
    return backbone.startswith('swin_')
    
def get_backbone_channels(backbone):
    """Lấy kích thước kênh đầu ra của backbone"""
    return BACKBONE_CHANNELS.get(backbone, [64, 128, 320, 512])

class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352))

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)


epsilon = 1e-7


def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))


def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        
        # Fix shape mismatch (new code)
        if logits.shape != label.shape:
            if logits.ndim == 5 and label.ndim == 4:
                logits = logits.squeeze(2)
            elif logits.ndim == 4 and label.ndim == 5:
                logits = logits.unsqueeze(2)
        
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss cho binary segmentation
    Rất hiệu quả cho medical image segmentation
    """

    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + self.smooth) / \
            (pred.sum() + target.sum() + self.smooth)

        return 1 - dice_coeff


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Cải tiến của Dice Loss
    Cho phép điều chỉnh trọng số cho False Positive và False Negative
    α = 0.5, β = 0.5 -> Dice Loss
    α > β -> Tập trung vào False Negative (tốt cho small objects)
    α < β -> Tập trung vào False Positive
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        true_pos = (pred * target).sum()
        false_neg = (target * (1 - pred)).sum()
        false_pos = ((1 - target) * pred).sum()

        tversky_coeff = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )

        return 1 - tversky_coeff


class ComboLoss(nn.Module):
    """
    Combo Loss = α * Dice Loss + (1-α) * BCE Loss
    Kết hợp ưu điểm của cả Dice và BCE
    """

    def __init__(self, alpha=0.5, smooth=1e-6, reduction='mean'):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth, reduction=reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target.float())
        return self.alpha * dice + (1 - self.alpha) * bce


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Tập trung vào ranh giới
    Sử dụng distance transform để tăng cường học ranh giới
    """

    def __init__(self, theta0=3, theta=5, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Tính gradient để detect ranh giới
        grad_pred = self._compute_gradient(pred)
        grad_target = self._compute_gradient(target)

        # Boundary loss
        boundary_loss = F.mse_loss(grad_pred, grad_target, reduction='none')

        # Weight boundary regions more
        boundary_weight = (grad_target > 0.1).float() * self.theta + 1.0
        weighted_loss = boundary_loss * boundary_weight

        if self.reduction == 'mean':
            return weighted_loss.mean()
        return weighted_loss

    def _compute_gradient(self, x):
        """Tính gradient magnitude cho boundary detection"""
        # Sobel operator
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        if x.dim() == 4:
            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)
        else:
            x = x.unsqueeze(1)
            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)
            grad_x = grad_x.squeeze(1)
            grad_y = grad_y.squeeze(1)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return gradient_magnitude


class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss - Kết hợp Asymmetric Focal Loss
    Cải tiến cho imbalanced segmentation
    """

    def __init__(self, weight=None, gamma=2., delta=0.6, reduction='mean'):
        super(UnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(),
                                                     weight=self.weight, reduction='none')

        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        # Asymmetric weighting
        alpha_t = target * self.delta + (1 - target) * (1 - self.delta)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Final loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceWeightedIoUFocalLoss(nn.Module):
    """
    Kết hợp Dice Loss + Weighted IoU + Focal Loss
    Tối ưu cho polyp segmentation với khả năng điều chỉnh trọng số từng thành phần
    
    Args:
        dice_weight (float): Trọng số cho Dice Loss (0.0-1.0)
        iou_weight (float): Trọng số cho Weighted IoU Loss (0.0-1.0) 
        focal_weight (float): Trọng số cho Focal Loss (0.0-1.0)
        focal_alpha (float): Alpha parameter cho Focal Loss
        focal_gamma (float): Gamma parameter cho Focal Loss
        smooth (float): Smoothing factor cho Dice và IoU
        spatial_weight (bool): Có sử dụng spatial weighting hay không
    """
    
    def __init__(self, 
                 dice_weight=0.4, 
                 iou_weight=0.3, 
                 focal_weight=0.3,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 smooth=1e-6,
                 spatial_weight=True,
                 reduction='mean'):
        super(DiceWeightedIoUFocalLoss, self).__init__()
        
        # Kiểm tra tổng trọng số
        total_weight = dice_weight + iou_weight + focal_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Cảnh báo: Tổng trọng số = {total_weight:.3f} ≠ 1.0")
            # Normalize weights
            dice_weight /= total_weight
            iou_weight /= total_weight  
            focal_weight /= total_weight
            print(f"Đã normalize: dice={dice_weight:.3f}, iou={iou_weight:.3f}, focal={focal_weight:.3f}")
            
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth
        self.spatial_weight = spatial_weight
        self.reduction = reduction
        
        # Initialize loss components
        self.focal_loss = FocalLossV1(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        
    def forward(self, pred, target):
        # Tính spatial weight nếu được kích hoạt
        if self.spatial_weight:
            # Enhanced boundary weighting
            spatial_weit = 1 + 5 * torch.abs(
                F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
            )
        else:
            spatial_weit = torch.ones_like(target)
            
        # 1. Dice Loss Component
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten tensors for dice calculation
        pred_flat = pred_sigmoid.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        spatial_flat = spatial_weit.contiguous().view(-1)
        
        # Weighted intersection and union for Dice
        intersection = (pred_flat * target_flat * spatial_flat).sum()
        dice_denominator = (pred_flat * spatial_flat).sum() + (target_flat * spatial_flat).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (dice_denominator + self.smooth)
        
        # 2. Weighted IoU Loss Component  
        union = pred_flat * spatial_flat + target_flat * spatial_flat - pred_flat * target_flat * spatial_flat
        iou_loss = 1 - (intersection + self.smooth) / (union.sum() + self.smooth)
        
        # 3. Focal Loss Component
        focal_loss_raw = self.focal_loss(pred, target)
        if self.spatial_weight:
            # Apply spatial weighting to focal loss
            focal_loss_weighted = (focal_loss_raw * spatial_weit).sum(dim=(2, 3)) / spatial_weit.sum(dim=(2, 3))
            focal_loss = focal_loss_weighted.mean()
        else:
            focal_loss = focal_loss_raw.mean()
        
        # Combine losses với trọng số đã định
        total_loss = (self.dice_weight * dice_loss + 
                     self.iou_weight * iou_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss
    
    def get_component_losses(self, pred, target):
        """Trả về loss của từng thành phần để theo dõi"""
        # Tính spatial weight
        if self.spatial_weight:
            spatial_weit = 1 + 5 * torch.abs(
                F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
            )
        else:
            spatial_weit = torch.ones_like(target)
            
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        spatial_flat = spatial_weit.contiguous().view(-1)
        
        # Component losses
        intersection = (pred_flat * target_flat * spatial_flat).sum()
        dice_denominator = (pred_flat * spatial_flat).sum() + (target_flat * spatial_flat).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (dice_denominator + self.smooth)
        
        union = pred_flat * spatial_flat + target_flat * spatial_flat - pred_flat * target_flat * spatial_flat
        iou_loss = 1 - (intersection + self.smooth) / (union.sum() + self.smooth)
        
        focal_loss_raw = self.focal_loss(pred, target)
        focal_loss = focal_loss_raw.mean()
        
        return {
            'dice_loss': dice_loss.item(),
            'iou_loss': iou_loss.item(), 
            'focal_loss': focal_loss.item(),
            'total_loss': (self.dice_weight * dice_loss + 
                          self.iou_weight * iou_loss + 
                          self.focal_weight * focal_loss).item()
        }


def get_loss_function(loss_type='structure', **kwargs):
    """
    Factory function để tạo loss function theo type
    """
    if loss_type == 'structure':
        return structure_loss
    elif loss_type == 'dice':
        return lambda pred, mask: DiceLoss(**kwargs)(pred, mask)
    elif loss_type == 'tversky':
        return lambda pred, mask: TverskyLoss(**kwargs)(pred, mask)
    elif loss_type == 'combo':
        return lambda pred, mask: ComboLoss(**kwargs)(pred, mask)
    elif loss_type == 'boundary':
        return lambda pred, mask: BoundaryLoss(**kwargs)(pred, mask)
    elif loss_type == 'unified_focal':
        return lambda pred, mask: UnifiedFocalLoss(**kwargs)(pred, mask)
    elif loss_type == 'focal':
        return lambda pred, mask: FocalLossV1(**kwargs)(pred, mask)
    elif loss_type == 'dice_weighted_iou_focal':
        return lambda pred, mask: DiceWeightedIoUFocalLoss(**kwargs)(pred, mask)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def structure_loss(pred, mask):
    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                  stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def enhanced_structure_loss(pred, mask, loss_type='structure', **loss_kwargs):
    """
    Enhanced structure loss với spatial weighting và custom loss function
    """
    # Spatial weighting - nhấn mạnh ranh giới
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    # Chọn loss function
    loss_fn = get_loss_function(loss_type, **loss_kwargs)

    if loss_type == 'structure':
        # Giữ nguyên structure loss gốc
        return structure_loss(pred, mask)
    else:
        # Áp dụng spatial weighting cho loss function khác
        base_loss = loss_fn(pred, mask)

        # Nếu là tensor không có spatial dimension, return trực tiếp
        if len(base_loss.shape) == 0:
            return base_loss

        # Weighted loss với spatial attention
        if len(base_loss.shape) >= 2:
            weighted_loss = (base_loss * weit).sum(dim=(-2, -1)
                                                   ) / weit.sum(dim=(-2, -1))
            return weighted_loss.mean()
        else:
            return base_loss.mean()


def generate_model_save_name(args):
    """Tự động tạo tên thư mục save model với đầy đủ thông tin"""
    model_name = get_model_name(args.backbone)
    
    # Format: ModelName_backbone_epochs_batchsize_loss_timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    save_name = f"{model_name}_{args.backbone}_{args.num_epochs}ep_{args.batchsize}bs_{args.loss_type}_{timestamp}"
    
    # Thêm thông tin loss parameters nếu có
    if args.loss_type == 'tversky':
        save_name += f"_a{args.tversky_alpha}_b{args.tversky_beta}"
    elif args.loss_type == 'combo':
        save_name += f"_a{args.combo_alpha}"
    elif args.loss_type == 'focal':
        save_name += f"_g{args.focal_gamma}"
    elif args.loss_type == 'unified_focal':
        save_name += f"_g{args.focal_gamma}_d{args.focal_delta}"
    
    return save_name


def create_training_metadata(args, model, epoch, total_step, loss_record, dice, iou):
    """Tạo metadata đầy đủ cho checkpoint"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metadata = {
        # Model information
        'model_name': get_model_name(args.backbone),
        'backbone': args.backbone,
        'architecture': 'UPerHead',
        'in_channels': [64, 128, 320, 512],
        'decoder_channels': 128,
        'num_classes': 1,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024,
        
        # Training configuration
        'num_epochs': args.num_epochs,
        'current_epoch': epoch,
        'completed': epoch >= args.num_epochs,
        'batch_size': args.batchsize,
        'init_lr': args.init_lr,
        'image_size': args.init_trainsize,
        'gradient_clip': args.clip,
        'total_steps_per_epoch': total_step,
        'multi_scale_training': [0.75, 1.0, 1.25],
        
        # Loss configuration
        'loss_type': args.loss_type,
        'loss_params': {},
        
        # Optimizer & Scheduler
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        
        # Dataset info
        'train_path': args.train_path,
        'train_save': args.train_save,
        
        # Training metadata
        'training_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mmseg_version': __version__,
        'resume_path': args.resume_path if args.resume_path else None,
        
        # Current metrics
        'last_metrics': {
            'epoch': epoch,
            'loss': float(loss_record.show()),
            'dice': float(dice.show()),
            'iou': float(iou.show())
        },
        
        # Training progress
        'training_completed': epoch >= args.num_epochs,
        'checkpoint_type': 'final' if epoch >= args.num_epochs else 'intermediate'
    }
    
    # Add loss-specific parameters
    if args.loss_type == 'tversky':
        metadata['loss_params'] = {
            'tversky_alpha': args.tversky_alpha,
            'tversky_beta': args.tversky_beta
        }
    elif args.loss_type == 'combo':
        metadata['loss_params'] = {
            'combo_alpha': args.combo_alpha
        }
    elif args.loss_type == 'focal':
        metadata['loss_params'] = {
            'focal_gamma': args.focal_gamma
        }
    elif args.loss_type == 'unified_focal':
        metadata['loss_params'] = {
            'focal_gamma': args.focal_gamma,
            'focal_delta': args.focal_delta
        }
    elif args.loss_type == 'dice_weighted_iou_focal':
        metadata['loss_params'] = {
            'dice_weight': args.dice_weight,
            'iou_weight': args.iou_weight,
            'focal_weight': args.focal_weight,
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma,
            'smooth': args.smooth,
            'spatial_weight': args.spatial_weight
        }
    elif args.loss_type == 'boundary':
        metadata['loss_params'] = {
            'boundary_theta0': getattr(args, 'boundary_theta0', 3),
            'boundary_theta': getattr(args, 'boundary_theta', 5)
        }
    
    return metadata


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    
    # Component loss tracking cho dice_weighted_iou_focal
    if args.loss_type == 'dice_weighted_iou_focal':
        dice_loss_record = AvgMeter()
        iou_loss_record = AvgMeter()
        focal_loss_record = AvgMeter()

    # Tracking time
    epoch_start_time = time.time()

    # Progress bar setup với tqdm
    print(f"\n==> Epoch {epoch}/{args.num_epochs}")
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}',
                        unit='batch', ncols=180,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    # Prepare loss function parameters
    loss_kwargs = {}
    if args.loss_type == 'tversky':
        loss_kwargs = {'alpha': args.tversky_alpha, 'beta': args.tversky_beta}
    elif args.loss_type == 'combo':
        loss_kwargs = {'alpha': args.combo_alpha}
    elif args.loss_type == 'unified_focal':
        loss_kwargs = {'gamma': args.focal_gamma, 'delta': args.focal_delta}
    elif args.loss_type == 'dice_weighted_iou_focal':
        loss_kwargs = {
            'dice_weight': args.dice_weight,
            'iou_weight': args.iou_weight,
            'focal_weight': args.focal_weight,
            'focal_alpha': args.focal_alpha,
            'focal_gamma': args.focal_gamma,
            'smooth': args.smooth,
            'spatial_weight': args.spatial_weight
        }

    with torch.autograd.set_detect_anomaly(False):  # Tắt anomaly detection
        for i, pack in enumerate(progress_bar, start=1):
            if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (
                    epoch * i) / (1.0 * total_step) * args.init_lr
            else:
                lr_scheduler.step()

            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                images = F.interpolate(images, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                map4, map3, map2, map1 = model(images)
                map1 = F.interpolate(map1, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                map2 = F.interpolate(map2, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                map3 = F.interpolate(map3, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)
                map4 = F.interpolate(map4, size=(
                    trainsize, trainsize), mode='bilinear', align_corners=True)

                # ---- loss calculation with chosen loss type ----
                loss = enhanced_structure_loss(map1, gts, args.loss_type, **loss_kwargs) + \
                    enhanced_structure_loss(map2, gts, args.loss_type, **loss_kwargs) + \
                    enhanced_structure_loss(map3, gts, args.loss_type, **loss_kwargs) + \
                    enhanced_structure_loss(
                        map4, gts, args.loss_type, **loss_kwargs)

                # ---- metrics ----
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
                
                # ---- track component losses cho dice_weighted_iou_focal ----
                if rate == 1 and args.loss_type == 'dice_weighted_iou_focal':
                    # Tính component losses từ map4 (main output)
                    loss_instance = DiceWeightedIoUFocalLoss(**loss_kwargs)
                    component_losses = loss_instance.get_component_losses(map4, gts)
                    
                    dice_loss_record.update(component_losses['dice_loss'], args.batchsize)
                    iou_loss_record.update(component_losses['iou_loss'], args.batchsize)
                    focal_loss_record.update(component_losses['focal_loss'], args.batchsize)
                
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

            # Update progress bar với metrics
            current_lr = optimizer.param_groups[0]['lr']
            
            if args.loss_type == 'dice_weighted_iou_focal':
                # Hiển thị chi tiết component losses
                progress_bar.set_postfix({
                    'Total': f'{loss_record.show():.4f}',
                    'Dice': f'{dice_loss_record.show():.4f}',
                    'IoU': f'{iou_loss_record.show():.4f}',
                    'Focal': f'{focal_loss_record.show():.4f}',
                    'Acc_D': f'{dice.show():.4f}',
                    'Acc_I': f'{iou.show():.4f}',
                    'LR': f'{current_lr:.6f}'
                })
            else:
                # Progress bar thông thường
                progress_bar.set_postfix({
                    'Loss': f'{loss_record.show():.4f}',
                    'Dice': f'{dice.show():.4f}',
                    'IoU': f'{iou.show():.4f}',
                    'LR': f'{current_lr:.6f}',
                    'Type': args.loss_type
                })

            # Final summary at end of epoch
            if i == total_step:
                progress_bar.close()
                epoch_time = time.time() - epoch_start_time
                logging.info(
                    f'\n{datetime.now()} Training Epoch [{epoch:03d}/{args.num_epochs:03d}] COMPLETED')
                logging.info(
                    f'Epoch time: {epoch_time:.2f}s ({epoch_time/60:.2f}m)')
                
                if args.loss_type == 'dice_weighted_iou_focal':
                    # Chi tiết component losses
                    logging.info(f'Final Loss Components:')
                    logging.info(f'  Total Loss: {loss_record.show():.4f}')
                    logging.info(f'  Dice Loss: {dice_loss_record.show():.4f} (weight: {args.dice_weight})')
                    logging.info(f'  IoU Loss: {iou_loss_record.show():.4f} (weight: {args.iou_weight})')
                    logging.info(f'  Focal Loss: {focal_loss_record.show():.4f} (weight: {args.focal_weight})')
                    logging.info(f'Final Accuracy - Dice: {dice.show():.4f}, IoU: {iou.show():.4f}')
                else:
                    logging.info(
                        f'Final metrics - Loss: {loss_record.show():.4f}, Dice: {dice.show():.4f}, IoU: {iou.show():.4f}')

    # Tạo checkpoint với metadata đầy đủ
    training_metadata = create_training_metadata(args, model, epoch, total_step, loss_record, dice, iou)
    
    # Thêm component loss information nếu sử dụng dice_weighted_iou_focal
    if args.loss_type == 'dice_weighted_iou_focal':
        training_metadata['component_losses'] = {
            'dice_loss': float(dice_loss_record.show()),
            'iou_loss': float(iou_loss_record.show()),
            'focal_loss': float(focal_loss_record.show()),
            'weights': {
                'dice_weight': args.dice_weight,
                'iou_weight': args.iou_weight,
                'focal_weight': args.focal_weight
            }
        }
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict(),
        'training_metadata': training_metadata
    }
    
    # Save checkpoint với tên mô tả
    if epoch == args.num_epochs:
        # Final checkpoint
        ckpt_path = save_path + 'final.pth'
        logging.info(f'[Saving Final Checkpoint:] {ckpt_path}')
    else:
        # Intermediate checkpoint (overwrite last.pth)
        ckpt_path = save_path + 'last.pth'
        logging.info(f'[Saving Checkpoint:] {ckpt_path}')
    
    torch.save(checkpoint, ckpt_path)
    
    # Also save epoch-specific checkpoint for final epoch
    if epoch == args.num_epochs:
        epoch_ckpt_path = save_path + f'epoch_{epoch}.pth'
        torch.save(checkpoint, epoch_ckpt_path)
        logging.info(f'[Also saved as:] {epoch_ckpt_path}')


# Thiết lập logging
def setup_train_logging(log_dir='logs', train_save='ColonFormerB3'):
    """Thiết lập logging system cho training"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{train_save}_{timestamp}.log')

    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


def log_train_model_info(model, args, total_step):
    """Log thông tin chi tiết về model và training setup"""
    logging.info("="*60)
    logging.info("MODEL & TRAINING INFORMATION")
    logging.info("="*60)

    # Thông tin cơ bản
    logging.info(f"Model: {get_model_name(args.backbone)}")
    logging.info(f"Backbone: MIT-{args.backbone}")
    logging.info(f"MMSeg Version: {__version__}")

    # Thông tin architecture
    logging.info(f"Decode Head: UPerHead")
    logging.info(f"Input Channels: [64, 128, 320, 512]")
    logging.info(f"Decoder Channels: 128")
    logging.info(f"Dropout Ratio: 0.1")
    logging.info(f"Number of Classes: 1 (Binary Segmentation)")

    # Loss function information
    logging.info(f"Loss Function: {args.loss_type.upper()}")
    if args.loss_type == 'structure':
        logging.info(f"  - Structure Loss (Focal + Weighted IoU)")
    elif args.loss_type == 'dice':
        logging.info(f"  - Dice Loss with spatial weighting")
    elif args.loss_type == 'tversky':
        logging.info(
            f"  - Tversky Loss (α={args.tversky_alpha}, β={args.tversky_beta})")
    elif args.loss_type == 'combo':
        logging.info(f"  - Combo Loss (α={args.combo_alpha}) = Dice + BCE")
    elif args.loss_type == 'boundary':
        logging.info(f"  - Boundary Loss with gradient-based weighting")
    elif args.loss_type == 'unified_focal':
        logging.info(
            f"  - Unified Focal Loss (γ={args.focal_gamma}, δ={args.focal_delta})")
    elif args.loss_type == 'dice_weighted_iou_focal':
        logging.info(f"  - Dice Weighted IoU Focal Loss")
        logging.info(f"    * Dice Weight: {args.dice_weight}")
        logging.info(f"    * IoU Weight: {args.iou_weight}")
        logging.info(f"    * Focal Weight: {args.focal_weight}")
        logging.info(f"    * Focal Alpha: {args.focal_alpha}")
        logging.info(f"    * Focal Gamma: {args.focal_gamma}")
        logging.info(f"    * Spatial Weight: {args.spatial_weight}")
    elif args.loss_type == 'focal':
        logging.info(f"  - Focal Loss (γ={args.focal_gamma})")

    # Thống kê parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    logging.info(f"Frozen Parameters: {frozen_params:,}")
    logging.info(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # Training configuration
    logging.info(f"\nTraining Configuration:")
    logging.info(f"  Epochs: {args.num_epochs}")
    logging.info(f"  Batch Size: {args.batchsize}")
    logging.info(f"  Initial LR: {args.init_lr}")
    logging.info(f"  Image Size: {args.init_trainsize}x{args.init_trainsize}")
    logging.info(f"  Gradient Clipping: {args.clip}")
    logging.info(f"  Steps per Epoch: {total_step}")
    logging.info(f"  Total Steps: {total_step * args.num_epochs}")
    logging.info(f"  Multi-scale Training: [0.75, 1.0, 1.25]")
    logging.info(f"  Optimizer: Adam")
    logging.info(f"  LR Scheduler: CosineAnnealingLR")
    logging.info(f"  Save Path: {args.train_save}")

    # Resume info
    if args.resume_path:
        logging.info(f"  Resume from: {args.resume_path}")
    else:
        logging.info(f"  Starting from: Scratch (Random weights)")

    logging.info("="*60)


def print_loss_recommendations():
    """In ra các gợi ý hệ số và câu lệnh cho các loss function"""
    print("\n" + "="*80)
    print(" DICE WEIGHTED IOU FOCAL LOSS - GOI Y HE SO VA CAU LENH")
    print("="*80)
    
    print("\nCAC HE SO GOI Y:")
    print("-" * 50)
    
    # Polyp nhỏ - nhấn mạnh Dice
    print("1. POLYP NHO (Small Polyps):")
    print("   dice_weight=0.5, iou_weight=0.2, focal_weight=0.3")
    print("   focal_alpha=0.25, focal_gamma=2.0")
    print("   -> Tăng Dice để capture small objects tốt hơn")
    
    # Polyp lớn - cân bằng
    print("\n2. POLYP LON (Large Polyps):")
    print("   dice_weight=0.4, iou_weight=0.3, focal_weight=0.3")
    print("   focal_alpha=0.25, focal_gamma=2.0")
    print("   -> Cân bằng cả 3 thành phần")
    
    # Boundary phức tạp
    print("\n3. RANH GIOI PHUC TAP (Complex Boundaries):")
    print("   dice_weight=0.3, iou_weight=0.4, focal_weight=0.3")
    print("   focal_alpha=0.25, focal_gamma=3.0")
    print("   -> Tăng IoU weight và focal gamma")
    
    # Imbalanced dataset
    print("\n4. DU LIEU IMBALANCED:")
    print("   dice_weight=0.4, iou_weight=0.25, focal_weight=0.35")
    print("   focal_alpha=0.5, focal_gamma=2.5")
    print("   -> Tăng focal weight và alpha")
    
    print("\nCAU LENH CHAY CHI TIET:")
    print("-" * 50)
    
    # Basic command
    print("\n1. CAU LENH CO BAN:")
    print("python train.py --loss_type dice_weighted_iou_focal --backbone b3 --num_epochs 20")
    
    # Small polyps
    print("\n2. TOAN CAU CHO POLYP NHO:")
    print("python train.py \\")
    print("  --loss_type dice_weighted_iou_focal \\")
    print("  --backbone b3 \\")
    print("  --num_epochs 25 \\")
    print("  --dice_weight 0.5 \\")
    print("  --iou_weight 0.2 \\")
    print("  --focal_weight 0.3 \\")
    print("  --focal_alpha 0.25 \\")
    print("  --focal_gamma 2.0 \\")
    print("  --batchsize 16 \\")
    print("  --init_lr 1e-4")
    
    # Complex boundaries
    print("\n3. TOAN CAU CHO RANH GIOI PHUC TAP:")
    print("python train.py \\")
    print("  --loss_type dice_weighted_iou_focal \\") 
    print("  --backbone b3 \\")
    print("  --num_epochs 30 \\")
    print("  --dice_weight 0.3 \\")
    print("  --iou_weight 0.4 \\")
    print("  --focal_weight 0.3 \\")
    print("  --focal_alpha 0.25 \\")
    print("  --focal_gamma 3.0 \\")
    print("  --batchsize 12 \\")
    print("  --init_lr 1e-4")
    
    # High performance setup
    print("\n4. CAU HINH HIEU SUAT CAO:")
    print("python train.py \\")
    print("  --loss_type dice_weighted_iou_focal \\")
    print("  --backbone b4 \\")
    print("  --num_epochs 40 \\")
    print("  --dice_weight 0.4 \\")
    print("  --iou_weight 0.3 \\")
    print("  --focal_weight 0.3 \\")
    print("  --focal_alpha 0.3 \\")
    print("  --focal_gamma 2.5 \\")
    print("  --batchsize 8 \\")
    print("  --init_lr 8e-5 \\")
    print("  --init_trainsize 384")
    
    # Imbalanced dataset
    print("\n5. DU LIEU IMBALANCED:")
    print("python train.py \\")
    print("  --loss_type dice_weighted_iou_focal \\")
    print("  --backbone b3 \\")
    print("  --num_epochs 35 \\")
    print("  --dice_weight 0.4 \\")
    print("  --iou_weight 0.25 \\")
    print("  --focal_weight 0.35 \\")
    print("  --focal_alpha 0.5 \\")
    print("  --focal_gamma 2.5 \\")
    print("  --batchsize 16")
    
    print("\nCHI TIET THAM SO:")
    print("-" * 50)
    print("dice_weight:   Trọng số Dice Loss (0.2-0.6) - Tốt cho small objects")
    print("iou_weight:    Trọng số IoU Loss (0.2-0.5) - Tốt cho shape accuracy")  
    print("focal_weight:  Trọng số Focal Loss (0.2-0.4) - Tốt cho hard examples")
    print("focal_alpha:   Alpha Focal (0.25-0.75) - Cân bằng pos/neg")
    print("focal_gamma:   Gamma Focal (1.5-3.0) - Focus on hard examples")
    print("smooth:        Smoothing factor (1e-8 to 1e-4)")
    print("spatial_weight: True/False - Có dùng spatial weighting không")
    
    print("\nKINH NGHIEM:")
    print("-" * 50)
    print("- Polyp nhỏ: Tăng dice_weight lên 0.5-0.6")
    print("- Boundary phức tạp: Tăng iou_weight và focal_gamma")
    print("- Dữ liệu imbalanced: Tăng focal_weight và focal_alpha")
    print("- Tăng focal_gamma để focus vào hard examples")
    print("- Giảm batch_size nếu model lớn (b4, b5)")
    print("- spatial_weight=True thường cho kết quả tốt hơn")
    
    print("="*80)

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type tversky --train_save tversky_epoch20

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type combo 

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type boundary  

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type unified_focal 

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type focal  

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type focal --ppm_scales 1,3,5,7 --train_save _focal_ppm-scale-1-3-5-7 

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type structure --ppm_scales 1,3,5,7 --train_save _structure_ppm-scale-1-3-5-7


# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type structure --ppm_scales 1,3,5,7 --train_save _structure_ppm-scale-1-3-5-7_cfp-4 --cfp_d 4

# python3 train.py --train_path ./Data/TrainDataset --backbone b3 --num_epochs 20 --batchsize 8 --init_lr 1e-4 --loss_type dice_weighted_iou_focal --ppm_scales 1,3,5,7 --train_save _dice_weighted_iou_focal_ppm-scale-1-3-5-7_cfp-4 --cfp_d 4


# -----------------------------------------

# python3 test.py --weight snapshots/ColonFormer-L_b3_20ep_8bs_boundary_20250720_1100/final.pth --test_path ./Data/TestDataset        

# python3 test.py --weight snapshots/ColonFormer-L_b3_20ep_8bs_combo_20250720_1004_a0.5/final.pth --test_path ./Data/TestDataset 

# python3 test.py --weight snapshots/ColonFormer-L_b3_20ep_8bs_focal_20250720_1241_g2.0/final.pth --test_path ./Data/TestDataset  

# python3 test.py --weight snapshots/ColonFormer-L_b3_20ep_8bs_unified_focal_20250720_1151_g2.0_d0.6/final.pth --test_path ./Data/TestDataset    

# python3 test.py --weight snapshots/_focal_ppm-scale-1-3-5-7/final.pth --test_path ./Data/TestDataset

# python3 test.py --weight snapshots/_structure_ppm-scale-1-3-5-7/final.pth --test_path ./Data/TestDataset

# python3 test.py --weight snapshots/_structure_ppm-scale-1-3-5-7_cfp-4/final.pth --test_path ./Data/TestDataset

# python3 test.py --weight snapshots/_dice_weighted_iou_focal_ppm-scale-1-3-5-7_cfp-4/final.pth --test_path ./Data/TestDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version (b1-b5, swin_tiny, swin_small, swin_base)')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='', help='custom save name (auto-generated if empty)')
    parser.add_argument('--resume_path', type=str,
                        default='', help='path to checkpoint for resume training')
    parser.add_argument('--loss_type', type=str,
                        default='structure', help='loss type')
    parser.add_argument('--tversky_alpha', type=float,
                        default=0.7, help='alpha for Tversky Loss')
    parser.add_argument('--tversky_beta', type=float,
                        default=0.3, help='beta for Tversky Loss')
    parser.add_argument('--combo_alpha', type=float,
                        default=0.5, help='alpha for Combo Loss')
    parser.add_argument('--focal_gamma', type=float,
                        default=2.0, help='gamma for Focal Loss')
    parser.add_argument('--focal_delta', type=float,
                        default=0.6, help='delta for Unified Focal Loss')
    parser.add_argument('--dice_weight', type=float,
                        default=0.4, help='dice_weight for Dice Weighted IoU Focal Loss')
    parser.add_argument('--iou_weight', type=float,
                        default=0.3, help='iou_weight for Dice Weighted IoU Focal Loss')
    parser.add_argument('--focal_weight', type=float,
                        default=0.3, help='focal_weight for Dice Weighted IoU Focal Loss')
    parser.add_argument('--focal_alpha', type=float,
                        default=0.25, help='focal_alpha for Dice Weighted IoU Focal Loss')
    parser.add_argument('--smooth', type=float,
                        default=1e-6, help='smooth for Dice Weighted IoU Focal Loss')
    parser.add_argument('--spatial_weight', type=bool,
                        default=True, help='spatial_weight for Dice Weighted IoU Focal Loss')
    # --- New CLI arguments for architecture tuning ---
    parser.add_argument('--ppm_scales', type=str,
                        default='1,2,3,6',
                        help='Comma-separated pooling scales for PPM module (e.g., "1,3,5,7")')
    parser.add_argument('--cfp_d', type=int,
                        default=8,
                        help='Dilation rate "d" for CFP modules (ColonFormer).')
    args = parser.parse_args()

    # Parse PPM scales into a tuple of ints once for reuse
    try:
        ppm_scales = tuple(int(s) for s in args.ppm_scales.split(',') if s.strip())
    except ValueError:
        raise ValueError(f"Invalid --ppm_scales value: {args.ppm_scales}. Provide comma-separated integers, e.g. '1,2,3,6'.")

    # Tự động tạo tên save nếu không được chỉ định
    if not args.train_save:
        args.train_save = generate_model_save_name(args)
        print(f"Auto-generated save name: {args.train_save}")

    # Setup logging
    log_file = setup_train_logging('logs', args.train_save)

    logging.info("="*60)
    logging.info("COLONFORMER TRAINING STARTED")
    logging.info("="*60)
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Auto-generated model save name: {args.train_save}")

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"Created save directory: {save_path}")
    else:
        logging.info(f"Save directory exists: {save_path}")

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob('{}/image/*'.format(args.train_path))
    train_mask_paths = glob('{}/mask/*'.format(args.train_path))
    train_img_paths.sort()
    train_mask_paths.sort()

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    total_step = len(train_loader)

    # Print dataset info
    logging.info(f"\n==> Dataset Information:")
    logging.info(f"Training images: {len(train_img_paths)}")
    logging.info(f"Training masks: {len(train_mask_paths)}")
    logging.info(f"Batch size: {args.batchsize}")
    logging.info(f"Total steps per epoch: {total_step}")
    logging.info(f"Image size: {args.init_trainsize}x{args.init_trainsize}")
    logging.info("Training from scratch without pretrained weights.")

    # Xác định loại backbone và tạo cấu hình tương ứng
    in_channels = get_backbone_channels(args.backbone)
    
    if is_swin_backbone(args.backbone):
        # Swin Transformer backbone
        swin_cfg = SWIN_CONFIGS[args.backbone]
        backbone_cfg = dict(
            type=args.backbone,  # swin_tiny, swin_small, swin_base
            pretrain_img_size=224,
            embed_dims=swin_cfg['embed_dims'],
            depths=swin_cfg['depths'],
            num_heads=swin_cfg['num_heads'],
            window_size=swin_cfg['window_size'],
            drop_path_rate=0.3,
            patch_norm=True,
        )
        pretrained_path = f'pretrained/{args.backbone}_patch4_window7_224.pth'
    else:
        # MiT backbone (mặc định)
        backbone_cfg = dict(
            type='mit_{}'.format(args.backbone),
            style='pytorch',
        )
        pretrained_path = f'pretrained/mit_{args.backbone}.pth'
    
    model = UNet(backbone=backbone_cfg,
        decode_head=dict(
        type='UPerHead',
        pool_scales=ppm_scales,  # use CLI-provided PPM scales
        in_channels=in_channels,  # Tự động điều chỉnh theo loại backbone
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        neck=None,
        auxiliary_head=None,
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
        pretrained=pretrained_path).cuda()

    # Log model and training information
    log_train_model_info(model, args, total_step)
    
    # Log chi tiết loss configuration nếu dùng dice_weighted_iou_focal
    if args.loss_type == 'dice_weighted_iou_focal':
        logging.info("\n" + "="*60)
        logging.info("DICE WEIGHTED IOU FOCAL LOSS CONFIGURATION")
        logging.info("="*60)
        logging.info(f"Component Weights:")
        logging.info(f"  Dice Weight: {args.dice_weight:.3f}")
        logging.info(f"  IoU Weight: {args.iou_weight:.3f}")
        logging.info(f"  Focal Weight: {args.focal_weight:.3f}")
        logging.info(f"  Total Weight: {args.dice_weight + args.iou_weight + args.focal_weight:.3f}")
        logging.info(f"Focal Parameters:")
        logging.info(f"  Focal Alpha: {args.focal_alpha}")
        logging.info(f"  Focal Gamma: {args.focal_gamma}")
        logging.info(f"Other Parameters:")
        logging.info(f"  Smoothing Factor: {args.smooth}")
        logging.info(f"  Spatial Weighting: {args.spatial_weight}")
        logging.info("="*60)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logging.info(f"\n==> Model Information:")
    logging.info(f"Backbone: MIT-{args.backbone}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=len(
                                                                  train_loader)*args.num_epochs,
                                                              eta_min=args.init_lr/1000)

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(
            f"Resumed training from epoch {start_epoch-1}, checkpoint: {args.resume_path}")

    logging.info("#"*20 + " Start Training " + "#"*20)
    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)
        
    logging.info("="*60)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info("="*60)
    logging.info(f"Final model saved in: {save_path}")
    logging.info(f"Final checkpoint: {save_path}final.pth")

    # Print loss recommendations
    print_loss_recommendations()
