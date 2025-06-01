import argparse
import logging
import os
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
                loss = structure_loss(map1, gts) + structure_loss(map2, gts) + \
                    structure_loss(map3, gts) + structure_loss(map4, gts)
                # with torch.autograd.set_detect_anomaly(True):
                # loss = nn.functional.binary_cross_entropy(map1, gts)
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
                'LR': f'{current_lr:.6f}'
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

    ckpt_path = save_path + 'last.pth'
    logging.info(f'[Saving Checkpoint:] {ckpt_path}')
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)


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
    logging.info(f"Model: ColonFormer")
    logging.info(f"Backbone: MIT-{args.backbone}")
    logging.info(f"MMSeg Version: {__version__}")

    # Thông tin architecture
    logging.info(f"Decode Head: UPerHead")
    logging.info(f"Input Channels: [64, 128, 320, 512]")
    logging.info(f"Decoder Channels: 128")
    logging.info(f"Dropout Ratio: 0.1")
    logging.info(f"Number of Classes: 1 (Binary Segmentation)")
    logging.info(f"Loss Function: Structure Loss (Focal + Weighted IoU)")

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
                        default=8, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='ConlonFormerB3')
    parser.add_argument('--resume_path', type=str,
                        default='', help='path to checkpoint for resume training')
    args = parser.parse_args()

    # Setup logging
    log_file = setup_train_logging('logs', args.train_save)

    logging.info("="*60)
    logging.info("COLONFORMER TRAINING STARTED")
    logging.info("="*60)
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(
            f"Resumed training from epoch {start_epoch}, checkpoint: {args.resume_path}")

    logging.info("#"*20 + " Start Training " + "#"*20)
    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)
