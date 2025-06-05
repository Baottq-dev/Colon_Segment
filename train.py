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
    'b1': 'ColonFormer-XS',
    'b2': 'ColonFormer-S', 
    'b3': 'ColonFormer-L',
    'b4': 'ColonFormer-XL',
    'b5': 'ColonFormer-XXL'
}

def get_model_name(backbone):
    """Lấy tên model từ backbone"""
    return BACKBONE_MODEL_MAPPING.get(backbone, f'ColonFormer-{backbone.upper()}')

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

    # Tracking time
    epoch_start_time = time.time()

    # Progress bar setup với tqdm
    print(f"\n==> Epoch {epoch}/{args.num_epochs}")
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}',
                        unit='batch', ncols=150,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    # Prepare loss function parameters
    loss_kwargs = {}
    if args.loss_type == 'tversky':
        loss_kwargs = {'alpha': args.tversky_alpha, 'beta': args.tversky_beta}
    elif args.loss_type == 'combo':
        loss_kwargs = {'alpha': args.combo_alpha}
    elif args.loss_type == 'unified_focal':
        loss_kwargs = {'gamma': args.focal_gamma, 'delta': args.focal_delta}

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
                logging.info(
                    f'Final metrics - Loss: {loss_record.show():.4f}, Dice: {dice.show():.4f}, IoU: {iou.show():.4f}')

    # Tạo checkpoint với metadata đầy đủ
    training_metadata = create_training_metadata(args, model, epoch, total_step, loss_record, dice, iou)
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=32, help='training batch size')
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
    args = parser.parse_args()

    # Tự động tạo tên save nếu không được chỉ định
    if not args.train_save:
        args.train_save = generate_model_save_name(args)
        print(f"🤖 Auto-generated save name: {args.train_save}")

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

    model = UNet(backbone=dict(
        type='mit_{}'.format(args.backbone),
        style='pytorch'),
        decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 320, 512],
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
        pretrained=None).cuda()

    # Log model and training information
    log_train_model_info(model, args, total_step)

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
