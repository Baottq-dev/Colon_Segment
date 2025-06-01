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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

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


def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))


def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    return intersection/(union+epsilon)

# Thiết lập logging


def setup_logging(log_dir='logs', test_dataset='all'):
    """Thiết lập logging system"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_{test_dataset}_{timestamp}.log')

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


def log_model_info(model, args):
    """Log thông tin chi tiết về model"""
    logging.info("="*60)
    logging.info("MODEL INFORMATION")
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
    logging.info(f"Loss Function: CrossEntropyLoss with Sigmoid")

    # Thống kê parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    logging.info(f"Frozen Parameters: {frozen_params:,}")
    logging.info(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # Thông tin weights
    if args.weight != '':
        logging.info(f"Loaded Weights: {args.weight}")
        try:
            checkpoint = torch.load(args.weight, map_location='cpu')
            if 'epoch' in checkpoint:
                logging.info(f"Checkpoint Epoch: {checkpoint['epoch']}")
            if 'state_dict' in checkpoint:
                logging.info(
                    f"State Dict Keys: {len(checkpoint['state_dict'])}")
        except:
            logging.warning(f"Cannot load checkpoint info from {args.weight}")
    else:
        logging.info("Weights: Random Initialization (No pretrained weights)")

    logging.info("="*60)


def create_metrics_visualization(results_dict, save_path='results'):
    """Tạo visualization cho metrics và lưu thành ảnh"""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Thiết lập style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ColonFormer - Test Results Summary',
                 fontsize=16, fontweight='bold')

    # 1. Bar chart cho metrics tổng quả
    metrics = ['Dice', 'mIoU', 'Precision', 'Recall']
    overall_values = [
        results_dict['overall']['dice'],
        results_dict['overall']['miou'],
        results_dict['overall']['precision'],
        results_dict['overall']['recall']
    ]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(metrics, overall_values, color=colors, alpha=0.8)
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)

    # Thêm giá trị lên bar
    for bar, value in zip(bars, overall_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Comparison across datasets
    if 'datasets' in results_dict and len(results_dict['datasets']) > 0:
        datasets = list(results_dict['datasets'].keys())
        dice_scores = [results_dict['datasets'][d]['dice'] for d in datasets]

        ax2.bar(datasets, dice_scores, color='#FF6B6B', alpha=0.7)
        ax2.set_title('Dice Score by Dataset', fontweight='bold')
        ax2.set_ylabel('Dice Score')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Thêm giá trị
        for i, (dataset, score) in enumerate(zip(datasets, dice_scores)):
            ax2.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'Single Dataset Test', ha='center',
                 va='center', transform=ax2.transAxes)
        ax2.set_title('Dataset Comparison', fontweight='bold')

    # 3. Metrics radar chart
    metrics_radar = ['Dice', 'mIoU', 'Precision', 'Recall']
    values_radar = overall_values + [overall_values[0]]  # Close the radar

    angles = np.linspace(0, 2 * np.pi, len(metrics_radar),
                         endpoint=False).tolist()
    angles += angles[:1]  # Close the radar

    ax3.plot(angles, values_radar, 'o-', linewidth=2, color='#FF6B6B')
    ax3.fill(angles, values_radar, alpha=0.25, color='#FF6B6B')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics_radar)
    ax3.set_ylim(0, 1)
    ax3.set_title('Performance Radar Chart', fontweight='bold')
    ax3.grid(True)

    # 4. Model info text
    ax4.axis('off')
    best_dataset = 'N/A'
    best_dice = 'N/A'

    if 'datasets' in results_dict and len(results_dict['datasets']) > 0:
        best_dataset = max(
            results_dict['datasets'], key=lambda x: results_dict['datasets'][x]['dice'])
        best_dice = f"{results_dict['datasets'][best_dataset]['dice']:.3f}"

    model_info = f"""
Model Information:
• Architecture: ColonFormer
• Backbone: MIT-{results_dict.get('backbone', 'b3')}
• Parameters: {results_dict.get('total_params', 'N/A'):,}
• Test Images: {results_dict.get('total_images', 'N/A')}
• Test Time: {results_dict.get('test_time', 'N/A'):.2f}s

Best Performance:
• Dataset: {best_dataset}
• Dice Score: {best_dice}

Overall Metrics:
• Dice: {overall_values[0]:.3f}
• mIoU: {overall_values[1]:.3f}  
• Precision: {overall_values[2]:.3f}
• Recall: {overall_values[3]:.3f}
"""

    ax4.text(0.05, 0.95, model_info, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Lưu ảnh
    img_path = os.path.join(save_path, f'test_results_{timestamp}.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    logging.info(f"Results visualization saved to: {img_path}")

    plt.close()

    return img_path


def get_scores(gts, prs):
    if len(gts) == 0:
        logging.warning("No test data found! Please check the test path.")
        return (0, 0, 0, 0)

    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)

    logging.info("="*50)
    logging.info("EVALUATION RESULTS")
    logging.info("="*50)
    logging.info(f"Dice Coefficient: {mean_dice:.6f}")
    logging.info(f"Mean IoU: {mean_iou:.6f}")
    logging.info(f"Precision: {mean_precision:.6f}")
    logging.info(f"Recall: {mean_recall:.6f}")
    logging.info("="*50)

    return (mean_iou, mean_dice, mean_precision, mean_recall)


def inference(model, args):
    start_time = time.time()

    logging.info("#"*60)
    logging.info("STARTING INFERENCE")
    logging.info("#"*60)

    model.eval()

    # Thử tìm dữ liệu test từ các dataset con
    test_datasets = ['Kvasir', 'ETIS-LaribPolypDB',
                     'CVC-ColonDB', 'CVC-ClinicDB', 'CVC-300']

    X_test = []
    y_test = []
    dataset_info = {}

    # Nếu chỉ định dataset cụ thể
    if args.test_dataset != 'all' and args.test_dataset in test_datasets:
        dataset_path = os.path.join(args.test_path, args.test_dataset)
        if os.path.exists(dataset_path):
            images_path = os.path.join(dataset_path, 'images', '*')
            masks_path = os.path.join(dataset_path, 'masks', '*')

            X_test = glob(images_path)
            y_test = glob(masks_path)

            if len(X_test) > 0:
                logging.info(
                    f"Testing on dataset: {args.test_dataset} with {len(X_test)} images")
                dataset_info[args.test_dataset] = len(X_test)
    else:
        # Kiểm tra xem có dataset nào trong test_path không
        for dataset in test_datasets:
            dataset_path = os.path.join(args.test_path, dataset)
            if os.path.exists(dataset_path):
                images_path = os.path.join(dataset_path, 'images', '*')
                masks_path = os.path.join(dataset_path, 'masks', '*')

                dataset_images = glob(images_path)
                dataset_masks = glob(masks_path)

                if len(dataset_images) > 0 and len(dataset_masks) > 0:
                    logging.info(
                        f"Found test dataset: {dataset} with {len(dataset_images)} images")
                    X_test.extend(dataset_images)
                    y_test.extend(dataset_masks)
                    dataset_info[dataset] = len(dataset_images)

    # Nếu không tìm thấy dữ liệu trong các dataset con, thử đường dẫn trực tiếp
    if len(X_test) == 0:
        X_test = glob('{}/images/*'.format(args.test_path))
        y_test = glob('{}/masks/*'.format(args.test_path))
        if len(X_test) > 0:
            dataset_info['custom'] = len(X_test)

    if len(X_test) == 0:
        logging.error(f"No test images found in {args.test_path}")
        logging.info("Available test datasets:")
        for dataset in test_datasets:
            dataset_path = os.path.join(args.test_path, dataset)
            if os.path.exists(dataset_path):
                logging.info(f"  - {dataset}")
        return

    X_test.sort()
    y_test.sort()

    logging.info(f"Total test images: {len(X_test)}")
    logging.info(f"Total test masks: {len(y_test)}")

    # Log dataset info
    for dataset, count in dataset_info.items():
        logging.info(f"  - {dataset}: {count} images")

    test_dataset = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    gts = []
    prs = []
    dataset_results = {}

    # Initialize dataset results
    for dataset in dataset_info.keys():
        dataset_results[dataset] = {'gts': [], 'prs': []}

    # Progress bar với tqdm
    progress_bar = tqdm(test_loader, desc='Testing', unit='img',
                        ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    current_idx = 0
    current_dataset = None
    dataset_bounds = {}

    # Calculate dataset boundaries
    start_idx = 0
    for dataset, count in dataset_info.items():
        dataset_bounds[dataset] = (start_idx, start_idx + count)
        start_idx += count

    with torch.no_grad():
        for i, pack in enumerate(progress_bar):
            # Determine current dataset
            for dataset, (start, end) in dataset_bounds.items():
                if start <= i < end:
                    current_dataset = dataset
                    break

            image, gt = pack
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res, res2, res3, res4 = model(image)
            res = F.interpolate(res, size=gt.shape,
                                mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pr = res.round()

            gts.append(gt)
            prs.append(pr)

            # Store per dataset
            if current_dataset and current_dataset in dataset_results:
                dataset_results[current_dataset]['gts'].append(gt)
                dataset_results[current_dataset]['prs'].append(pr)

            # Update progress bar
            progress_bar.set_postfix({
                'Dataset': current_dataset or 'Unknown',
                'Progress': f'{i+1}/{len(test_loader)}'
            })

    progress_bar.close()

    # Calculate overall metrics
    overall_metrics = get_scores(gts, prs)

    # Calculate per-dataset metrics
    dataset_metrics = {}
    for dataset, data in dataset_results.items():
        if len(data['gts']) > 0:
            metrics = get_scores(data['gts'], data['prs'])
            dataset_metrics[dataset] = {
                'miou': metrics[0],
                'dice': metrics[1],
                'precision': metrics[2],
                'recall': metrics[3],
                'images': len(data['gts'])
            }
            logging.info(f"\n{dataset} Dataset Results:")
            logging.info(f"  Images: {len(data['gts'])}")
            logging.info(f"  Dice: {metrics[1]:.6f}")
            logging.info(f"  mIoU: {metrics[0]:.6f}")
            logging.info(f"  Precision: {metrics[2]:.6f}")
            logging.info(f"  Recall: {metrics[3]:.6f}")

    test_time = time.time() - start_time
    logging.info(f"\nTotal inference time: {test_time:.2f} seconds")
    logging.info(
        f"Average time per image: {test_time/len(X_test):.4f} seconds")

    # Prepare results dictionary for visualization
    results_dict = {
        'overall': {
            'miou': overall_metrics[0],
            'dice': overall_metrics[1],
            'precision': overall_metrics[2],
            'recall': overall_metrics[3]
        },
        'datasets': dataset_metrics,
        'backbone': args.backbone,
        'total_params': sum(p.numel() for p in model.parameters()),
        'total_images': len(X_test),
        'test_time': test_time
    }

    # Create and save visualization
    img_path = create_metrics_visualization(results_dict)

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str,
                        default='b3')
    parser.add_argument('--weight', type=str,
                        default='')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset', help='path to dataset')
    parser.add_argument('--test_dataset', type=str,
                        default='all', help='specific test dataset: Kvasir, ETIS-LaribPolypDB, CVC-ColonDB, CVC-ClinicDB, CVC-300, or all')
    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging('logs', args.test_dataset)
    logging.info(f"Log file created: {log_file}")

    logging.info("="*60)
    logging.info("COLONFORMER TESTING STARTED")
    logging.info("="*60)
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Test dataset: {args.test_dataset}")
    logging.info(f"Test path: {args.test_path}")
    logging.info(f"Backbone: {args.backbone}")
    logging.info(
        f"Weight file: {args.weight if args.weight else 'None (Random weights)'}")

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

    # Log model information
    log_model_info(model, args)

    if args.weight != '':
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Successfully loaded weights from: {args.weight}")

    # Run inference
    results = inference(model, args)

    logging.info("="*60)
    logging.info("TESTING COMPLETED SUCCESSFULLY")
    logging.info("="*60)
    logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if results:
        logging.info("Final Summary:")
        logging.info(f"  Overall Dice: {results['overall']['dice']:.6f}")
        logging.info(f"  Overall mIoU: {results['overall']['miou']:.6f}")
        logging.info(f"  Total images tested: {results['total_images']}")
        logging.info(f"  Total test time: {results['test_time']:.2f}s")
