import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mmseg.models.segmentors import ColonFormer as UNet
from mmseg import __version__
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import time
import sys
import random
import logging
import argparse

# Import loss functions từ train.py
try:
    from train import (
        DiceWeightedIoUFocalLoss,
        get_loss_function,
        dice_m,
        iou_m,
        precision_m,
        recall_m
    )
    HAS_LOSS_FUNCTIONS = True
except ImportError:
    HAS_LOSS_FUNCTIONS = False
    logging.warning("Cannot import loss functions from train.py")


try:
    from scipy import stats
except ImportError:
    stats = None
    logging.warning(
        "scipy not available, confidence intervals will not be calculated")


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, trainsize=352):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.trainsize = trainsize

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (self.trainsize, self.trainsize))
        mask = cv2.resize(mask, (self.trainsize, self.trainsize))

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


def calculate_extended_metrics(y_true, y_pred):
    """Calculate extended metrics including accuracy, sensitivity, specificity"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Threshold predictions
    y_pred_binary = (y_pred_flat > 0.5).astype(np.float32)
    y_true_binary = (y_true_flat > 0.5).astype(np.float32)

    # Calculate confusion matrix components
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)  # Same as recall
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)

    # F1-score
    f1_score = 2 * (precision * sensitivity) / \
        (precision + sensitivity + epsilon)

    # IoU
    iou = tp / (tp + fp + fn + epsilon)

    # Dice
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': sensitivity,
        'f1_score': f1_score,
        'iou': iou,
        'dice': dice,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def calculate_confidence_intervals(scores, confidence=0.95):
    """Calculate confidence intervals for metrics"""
    if stats is None:
        return None

    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    se = std / np.sqrt(n)

    # t-distribution critical value
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_critical * se

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': mean - margin_error,
        'ci_upper': mean + margin_error,
        'margin_error': margin_error
    }


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
    logging.info(f"Model: {get_model_name(args.backbone)}")
    logging.info(f"Backbone: {args.backbone}")
    logging.info(f"MMSeg Version: {__version__}")

    # Thông tin architecture
    logging.info(f"Decode Head: UPerHead")
    logging.info(f"Input Channels: {get_backbone_channels(args.backbone)}")
    logging.info(f"Decoder Channels: 128")
    logging.info(f"Dropout Ratio: 0.1")
    logging.info(f"Number of Classes: 1 (Binary Segmentation)")

    # Loss function information (if available from training)
    if hasattr(args, 'loss_type'):
        logging.info(f"Training Loss Function: {args.loss_type.upper()}")
        if args.loss_type == 'structure':
            logging.info(f"  - Structure Loss (Focal + Weighted IoU)")
        elif args.loss_type == 'dice':
            logging.info(f"  - Dice Loss with spatial weighting")
        elif args.loss_type == 'tversky':
            tversky_alpha = getattr(args, 'tversky_alpha', 0.7)
            tversky_beta = getattr(args, 'tversky_beta', 0.3)
            logging.info(
                f"  - Tversky Loss (α={tversky_alpha}, β={tversky_beta})")
        elif args.loss_type == 'combo':
            combo_alpha = getattr(args, 'combo_alpha', 0.5)
            logging.info(f"  - Combo Loss (α={combo_alpha}) = Dice + BCE")
        elif args.loss_type == 'boundary':
            logging.info(f"  - Boundary Loss with gradient-based weighting")
        elif args.loss_type == 'unified_focal':
            focal_gamma = getattr(args, 'focal_gamma', 2.0)
            focal_delta = getattr(args, 'focal_delta', 0.6)
            logging.info(
                f"  - Unified Focal Loss (γ={focal_gamma}, δ={focal_delta})")
        elif args.loss_type == 'focal':
            focal_gamma = getattr(args, 'focal_gamma', 2.0)
            logging.info(f"  - Focal Loss (γ={focal_gamma})")
    else:
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
    fig.suptitle(f'{get_model_name(results_dict.get("backbone", "b3"))} - Test Results Summary',
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
• Architecture: {get_model_name(results_dict.get('backbone', 'b3'))}
• Backbone: {results_dict.get('backbone', 'b3')}
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
        return (0, 0, 0, 0), {}, []

    # Calculate metrics for each image
    individual_metrics = []
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    f1_scores = []
    sensitivity_scores = []
    specificity_scores = []

    for gt, pr in zip(gts, prs):
        # Original metrics
        precision = precision_np(gt, pr)
        recall = recall_np(gt, pr)
        iou = iou_np(gt, pr)
        dice = dice_np(gt, pr)
        
        # Extended metrics
        extended = calculate_extended_metrics(gt, pr)
        
        individual_metrics.append({
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'accuracy': extended['accuracy'],
            'f1_score': extended['f1_score'],
            'sensitivity': extended['sensitivity'],
            'specificity': extended['specificity']
        })
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(extended['accuracy'])
        f1_scores.append(extended['f1_score'])
        sensitivity_scores.append(extended['sensitivity'])
        specificity_scores.append(extended['specificity'])

    # Calculate mean metrics
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_f1 = np.mean(f1_scores)
    mean_sensitivity = np.mean(sensitivity_scores)
    mean_specificity = np.mean(specificity_scores)

    # Calculate statistics
    stats_dict = {
        'dice': {
            'mean': mean_dice,
            'std': np.std(dice_scores),
            'min': np.min(dice_scores),
            'max': np.max(dice_scores),
            'median': np.median(dice_scores),
            'q25': np.percentile(dice_scores, 25),
            'q75': np.percentile(dice_scores, 75)
        },
        'iou': {
            'mean': mean_iou,
            'std': np.std(iou_scores),
            'min': np.min(iou_scores),
            'max': np.max(iou_scores),
            'median': np.median(iou_scores),
            'q25': np.percentile(iou_scores, 25),
            'q75': np.percentile(iou_scores, 75)
        },
        'precision': {
            'mean': mean_precision,
            'std': np.std(precision_scores),
            'min': np.min(precision_scores),
            'max': np.max(precision_scores),
            'median': np.median(precision_scores),
            'q25': np.percentile(precision_scores, 25),
            'q75': np.percentile(precision_scores, 75)
        },
        'recall': {
            'mean': mean_recall,
            'std': np.std(recall_scores),
            'min': np.min(recall_scores),
            'max': np.max(recall_scores),
            'median': np.median(recall_scores),
            'q25': np.percentile(recall_scores, 25),
            'q75': np.percentile(recall_scores, 75)
        },
        'accuracy': {
            'mean': mean_accuracy,
            'std': np.std(accuracy_scores),
            'min': np.min(accuracy_scores),
            'max': np.max(accuracy_scores),
            'median': np.median(accuracy_scores),
            'q25': np.percentile(accuracy_scores, 25),
            'q75': np.percentile(accuracy_scores, 75)
        },
        'f1_score': {
            'mean': mean_f1,
            'std': np.std(f1_scores),
            'min': np.min(f1_scores),
            'max': np.max(f1_scores),
            'median': np.median(f1_scores),
            'q25': np.percentile(f1_scores, 25),
            'q75': np.percentile(f1_scores, 75)
        },
        'sensitivity': {
            'mean': mean_sensitivity,
            'std': np.std(sensitivity_scores),
            'min': np.min(sensitivity_scores),
            'max': np.max(sensitivity_scores),
            'median': np.median(sensitivity_scores),
            'q25': np.percentile(sensitivity_scores, 25),
            'q75': np.percentile(sensitivity_scores, 75)
        },
        'specificity': {
            'mean': mean_specificity,
            'std': np.std(specificity_scores),
            'min': np.min(specificity_scores),
            'max': np.max(specificity_scores),
            'median': np.median(specificity_scores),
            'q25': np.percentile(specificity_scores, 25),
            'q75': np.percentile(specificity_scores, 75)
        }
    }

    return (mean_iou, mean_dice, mean_precision, mean_recall), stats_dict, individual_metrics


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
    
    # Component loss tracking cho dice_weighted_iou_focal
    component_losses = []
    has_component_loss = HAS_LOSS_FUNCTIONS and args.loss_type == 'dice_weighted_iou_focal'
    
    if has_component_loss:
        logging.info(f"Tracking component losses for dice_weighted_iou_focal")
        logging.info(f"  Dice weight: {args.dice_weight}")
        logging.info(f"  IoU weight: {args.iou_weight}")
        logging.info(f"  Focal weight: {args.focal_weight}")

    # Initialize dataset results
    for dataset in dataset_info.keys():
        dataset_results[dataset] = {'gts': [], 'prs': []}
        if has_component_loss:
            dataset_results[dataset]['component_losses'] = []

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
            
            # Calculate component losses nếu dùng dice_weighted_iou_focal
            if has_component_loss:
                # Prepare loss kwargs
                loss_kwargs = {
                    'dice_weight': args.dice_weight,
                    'iou_weight': args.iou_weight,
                    'focal_weight': args.focal_weight,
                    'focal_alpha': args.focal_alpha,
                    'focal_gamma': args.focal_gamma,
                    'smooth': args.smooth,
                    'spatial_weight': args.spatial_weight
                }
                
                # Calculate component losses
                gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda()
                res_tensor = res.unsqueeze(0)
                
                loss_instance = DiceWeightedIoUFocalLoss(**loss_kwargs)
                comp_loss = loss_instance.get_component_losses(res_tensor, gt_tensor)
                component_losses.append(comp_loss)
                
                # Store per dataset component loss
                if current_dataset and current_dataset in dataset_results:
                    dataset_results[current_dataset]['component_losses'].append(comp_loss)
            
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
    overall_metrics, stats_dict, individual_metrics = get_scores(gts, prs)

    # Calculate per-dataset metrics
    dataset_metrics = {}
    dataset_individual_metrics = {}

    for dataset, data in dataset_results.items():
        if len(data['gts']) > 0:
            metrics, dataset_stats, dataset_individual = get_scores(
                data['gts'], data['prs'])
            
            # Base metrics
            dataset_metrics[dataset] = {
                'miou': metrics[0],
                'dice': metrics[1],
                'precision': metrics[2],
                'recall': metrics[3],
                'images': len(data['gts']),
                # Add extended metrics from stats
                'accuracy': dataset_stats['accuracy']['mean'],
                'f1_score': dataset_stats['f1_score']['mean'],
                'sensitivity': dataset_stats['sensitivity']['mean'],
                'specificity': dataset_stats['specificity']['mean']
            }
            
            # Add component loss statistics if available
            if has_component_loss and 'component_losses' in data and len(data['component_losses']) > 0:
                comp_losses = data['component_losses']
                avg_dice_loss = np.mean([cl['dice_loss'] for cl in comp_losses])
                avg_iou_loss = np.mean([cl['iou_loss'] for cl in comp_losses])
                avg_focal_loss = np.mean([cl['focal_loss'] for cl in comp_losses])
                avg_total_loss = np.mean([cl['total_loss'] for cl in comp_losses])
                
                dataset_metrics[dataset]['component_losses'] = {
                    'dice_loss': avg_dice_loss,
                    'iou_loss': avg_iou_loss,
                    'focal_loss': avg_focal_loss,
                    'total_loss': avg_total_loss,
                    'dice_contribution': avg_dice_loss * args.dice_weight,
                    'iou_contribution': avg_iou_loss * args.iou_weight,
                    'focal_contribution': avg_focal_loss * args.focal_weight,
                    'weights': {
                        'dice_weight': args.dice_weight,
                        'iou_weight': args.iou_weight,
                        'focal_weight': args.focal_weight
                    }
                }
            
            dataset_individual_metrics[dataset] = dataset_individual

    # Print comprehensive results
    if dataset_metrics and stats_dict:
        print_comprehensive_results(dataset_metrics, overall_metrics, stats_dict, {
            'model_name': get_model_name(args.backbone),
            'backbone': args.backbone,
            'total_images': len(X_test),
            'total_params': sum(p.numel() for p in model.parameters()),
            'test_time': time.time() - start_time
        })
        
        # Save results to accumulated database
        save_test_results(dataset_metrics, overall_metrics, stats_dict, {
            'model_name': get_model_name(args.backbone),
            'backbone': args.backbone,
            'total_images': len(X_test),
            'total_params': sum(p.numel() for p in model.parameters()),
            'test_time': time.time() - start_time
        }, args)

    # Calculate overall component loss statistics
    overall_component_losses = None
    if has_component_loss and len(component_losses) > 0:
        overall_avg_dice_loss = np.mean([cl['dice_loss'] for cl in component_losses])
        overall_avg_iou_loss = np.mean([cl['iou_loss'] for cl in component_losses])
        overall_avg_focal_loss = np.mean([cl['focal_loss'] for cl in component_losses])
        overall_avg_total_loss = np.mean([cl['total_loss'] for cl in component_losses])
        
        overall_component_losses = {
            'dice_loss': overall_avg_dice_loss,
            'iou_loss': overall_avg_iou_loss,
            'focal_loss': overall_avg_focal_loss,
            'total_loss': overall_avg_total_loss,
            'dice_contribution': overall_avg_dice_loss * args.dice_weight,
            'iou_contribution': overall_avg_iou_loss * args.iou_weight,
            'focal_contribution': overall_avg_focal_loss * args.focal_weight,
            'weights': {
                'dice_weight': args.dice_weight,
                'iou_weight': args.iou_weight,
                'focal_weight': args.focal_weight
            }
        }

    # Prepare results dictionary for visualization
    results_dict = {
        'overall': {
            'miou': overall_metrics[0],
            'dice': overall_metrics[1],
            'precision': overall_metrics[2],
            'recall': overall_metrics[3]
        },
        'datasets': dataset_metrics,
        'stats': stats_dict,
        'individual_metrics': individual_metrics,
        'backbone': args.backbone,
        'total_params': sum(p.numel() for p in model.parameters()),
        'total_images': len(X_test),
        'test_time': time.time() - start_time,
        'loss_type': args.loss_type
    }
    
    # Add component loss info if available
    if overall_component_losses:
        results_dict['overall']['component_losses'] = overall_component_losses

    # Create and save visualization
    img_path = create_metrics_visualization(results_dict)

    return results_dict


def print_comprehensive_results(dataset_metrics, overall_metrics, stats_dict, test_info):
    """In kết quả toàn diện và dễ đọc"""
    
    # Header với thông tin test
    print("\n" + "="*100)
    print(f"{'COLONFORMER TEST RESULTS':^100}")
    print("="*100)
    
    print(f"Model: {test_info['model_name']:<20} | Total Images: {test_info['total_images']:<6} | Test Time: {test_info['test_time']:.2f}s")
    print(f"Backbone: {test_info['backbone']:<15} | Parameters: {test_info['total_params']:}")
    print("-"*100)
    
    # Dataset results table
    print(f"\n{'DATASET PERFORMANCE SUMMARY':^100}")
    print("="*100)
    print(f"{'Dataset':<18} | {'Images':<6} | {'Dice':<6} | {'mIoU':<6} | {'Prec':<6} | {'Recall':<6} | {'Acc':<6} | {'F1':<6} | {'Performance'}")
    print("-"*100)
    
    # Sort datasets by dice score
    sorted_datasets = sorted(dataset_metrics.items(), key=lambda x: x[1]['dice'], reverse=True)
    
    for i, (dataset, metrics) in enumerate(sorted_datasets, 1):
        # Performance level
        dice = metrics['dice']
        if dice >= 0.8:
            perf_level = "EXCELLENT"
        elif dice >= 0.7:
            perf_level = "GOOD"
        elif dice >= 0.6:
            perf_level = "FAIR"
        elif dice >= 0.4:
            perf_level = "POOR"
        else:
            perf_level = "VERY POOR"
            
        print(f"{dataset:<18} | {metrics['images']:<6} | {dice:<6.3f} | {metrics['miou']:<6.3f} | "
              f"{metrics['precision']:<6.3f} | {metrics['recall']:<6.3f} | {metrics['accuracy']:<6.3f} | "
              f"{metrics['f1_score']:<6.3f} | {perf_level}")
    
    print("-"*100)
    
    # Overall performance
    dice_mean = stats_dict['dice']['mean']
    dice_std = stats_dict['dice']['std']
    
    if dice_mean >= 0.8:
        overall_perf = "EXCELLENT"
    elif dice_mean >= 0.7:
        overall_perf = "GOOD"
    elif dice_mean >= 0.6:
        overall_perf = "FAIR"
    elif dice_mean >= 0.4:
        overall_perf = "POOR"
    else:
        overall_perf = "VERY POOR"
    
    print(f"{'OVERALL':<18} | {test_info['total_images']:<6} | {dice_mean:<6.3f} | {stats_dict['iou']['mean']:<6.3f} | "
          f"{stats_dict['precision']['mean']:<6.3f} | {stats_dict['recall']['mean']:<6.3f} | "
          f"{stats_dict['accuracy']['mean']:<6.3f} | {stats_dict['f1_score']['mean']:<6.3f} | {overall_perf}")
    
    print("="*100)
    
    # Key insights
    print(f"\n{'KEY INSIGHTS':^100}")
    print("="*100)
    
    best_dataset = max(dataset_metrics, key=lambda x: dataset_metrics[x]['dice'])
    worst_dataset = min(dataset_metrics, key=lambda x: dataset_metrics[x]['dice'])
    
    print(f"Best Dataset:     {best_dataset} (Dice: {dataset_metrics[best_dataset]['dice']:.3f})")
    print(f"Worst Dataset:    {worst_dataset} (Dice: {dataset_metrics[worst_dataset]['dice']:.3f})")
    
    dice_gap = dataset_metrics[best_dataset]['dice'] - dataset_metrics[worst_dataset]['dice']
    print(f"Performance Gap:  {dice_gap:.3f} ({dice_gap*100:.1f}%)")
    
    # Model characteristics
    precision_mean = stats_dict['precision']['mean']
    recall_mean = stats_dict['recall']['mean']
    
    if precision_mean > recall_mean + 0.05:
        model_char = "Conservative (High Precision, Lower Recall)"
    elif recall_mean > precision_mean + 0.05:
        model_char = "Sensitive (High Recall, Lower Precision)"
    else:
        model_char = "Balanced (Similar Precision & Recall)"
    
    print(f"Model Character:  {model_char}")
    print(f"Stability:        σ = {dice_std:.3f} ({'Stable' if dice_std < 0.1 else 'Variable'})")
    
    # Performance distribution
    print(f"\n{'PERFORMANCE DISTRIBUTION':^100}")
    print("="*100)
    
    excellent_count = sum(1 for m in dataset_metrics.values() if m['dice'] >= 0.8)
    good_count = sum(1 for m in dataset_metrics.values() if 0.7 <= m['dice'] < 0.8)
    fair_count = sum(1 for m in dataset_metrics.values() if 0.6 <= m['dice'] < 0.7)
    poor_count = sum(1 for m in dataset_metrics.values() if 0.4 <= m['dice'] < 0.6)
    very_poor_count = sum(1 for m in dataset_metrics.values() if m['dice'] < 0.4)
    
    total_datasets = len(dataset_metrics)
    
    print(f"Excellent (≥0.8): {excellent_count:2d}/{total_datasets} ({excellent_count/total_datasets*100:5.1f}%)")
    print(f"Good     (≥0.7): {good_count:2d}/{total_datasets} ({good_count/total_datasets*100:5.1f}%)")
    print(f"Fair     (≥0.6): {fair_count:2d}/{total_datasets} ({fair_count/total_datasets*100:5.1f}%)")
    print(f"Poor     (≥0.4): {poor_count:2d}/{total_datasets} ({poor_count/total_datasets*100:5.1f}%)")
    print(f"Very Poor (<0.4): {very_poor_count:2d}/{total_datasets} ({very_poor_count/total_datasets*100:5.1f}%)")
    
    print("="*100)
    
    # Recommendations based on performance
    print(f"\n{'RECOMMENDATIONS':^100}")
    print("="*100)
    
    if dice_mean < 0.5:
        print("CRITICAL: Very low performance detected!")
        print("   - Check if model weights are loaded correctly")
        print("   - Verify data preprocessing matches training")
        print("   - Consider retraining with better hyperparameters")
    elif dice_mean < 0.7:
        print("MODERATE: Performance below expectations")
        print("   - Consider data augmentation strategies")
        print("   - Experiment with different loss functions")
        print("   - Check for dataset distribution mismatch")
    else:
        print("GOOD: Model performance is acceptable")
        print("   - Consider fine-tuning on challenging datasets")
        print("   - Experiment with ensemble methods")
    
    if dice_std > 0.15:
        print("HIGH VARIANCE: Model performance is inconsistent")
        print("   - Consider more robust training strategies")
        print("   - Implement cross-validation")
    
    # Component Loss Analysis (nếu có)
    has_component_loss = any('component_losses' in metrics for metrics in dataset_metrics.values())
    
    if has_component_loss:
        print(f"\n{'COMPONENT LOSS ANALYSIS':^100}")
        print("="*100)
        
        # Tìm một dataset có component loss để lấy weights
        sample_dataset = next((dataset for dataset, metrics in dataset_metrics.items() 
                             if 'component_losses' in metrics), None)
        
        if sample_dataset:
            weights = dataset_metrics[sample_dataset]['component_losses']['weights']
            print(f"Loss Weights: Dice={weights['dice_weight']:.2f} | IoU={weights['iou_weight']:.2f} | Focal={weights['focal_weight']:.2f}")
            print("-"*100)
            
            print(f"{'Dataset':<18} | {'Total Loss':<10} | {'Dice Loss':<10} | {'IoU Loss':<10} | {'Focal Loss':<10} | {'Contributions'}")
            print("-"*100)
            
            for dataset, metrics in sorted_datasets:
                if 'component_losses' in metrics:
                    cl = metrics['component_losses']
                    contrib_str = f"D:{cl['dice_contribution']:.3f} I:{cl['iou_contribution']:.3f} F:{cl['focal_contribution']:.3f}"
                    print(f"{dataset:<18} | {cl['total_loss']:<10.4f} | {cl['dice_loss']:<10.4f} | {cl['iou_loss']:<10.4f} | {cl['focal_loss']:<10.4f} | {contrib_str}")
            
            # Overall component loss statistics
            print("-"*100)
            avg_dice_loss = np.mean([metrics['component_losses']['dice_loss'] 
                                   for metrics in dataset_metrics.values() 
                                   if 'component_losses' in metrics])
            avg_iou_loss = np.mean([metrics['component_losses']['iou_loss'] 
                                  for metrics in dataset_metrics.values() 
                                  if 'component_losses' in metrics])
            avg_focal_loss = np.mean([metrics['component_losses']['focal_loss'] 
                                    for metrics in dataset_metrics.values() 
                                    if 'component_losses' in metrics])
            avg_total_loss = np.mean([metrics['component_losses']['total_loss'] 
                                    for metrics in dataset_metrics.values() 
                                    if 'component_losses' in metrics])
            
            contrib_dice = avg_dice_loss * weights['dice_weight']
            contrib_iou = avg_iou_loss * weights['iou_weight']
            contrib_focal = avg_focal_loss * weights['focal_weight']
            
            print(f"{'AVERAGE':<18} | {avg_total_loss:<10.4f} | {avg_dice_loss:<10.4f} | {avg_iou_loss:<10.4f} | {avg_focal_loss:<10.4f} | D:{contrib_dice:.3f} I:{contrib_iou:.3f} F:{contrib_focal:.3f}")
            
            # Component loss insights
            print(f"\n{'COMPONENT LOSS INSIGHTS':^100}")
            print("="*100)
            
            # Tìm thành phần loss dominant
            if contrib_dice > contrib_iou and contrib_dice > contrib_focal:
                dominant = "Dice Loss"
                insight = "Model struggles most with overlap/intersection accuracy"
            elif contrib_iou > contrib_dice and contrib_iou > contrib_focal:
                dominant = "IoU Loss"
                insight = "Model struggles most with shape/boundary precision"
            else:
                dominant = "Focal Loss"
                insight = "Model struggles most with hard examples/class imbalance"
            
            print(f"Dominant Component: {dominant}")
            print(f"Primary Challenge: {insight}")
            
            # Recommendation dựa trên component loss
            if avg_dice_loss > 0.5:
                print("Dice Loss High: Consider increasing dice_weight or improving overlap accuracy")
            if avg_iou_loss > 0.5:
                print("IoU Loss High: Consider increasing iou_weight or improving boundary precision")
            if avg_focal_loss > 0.5:
                print("Focal Loss High: Consider increasing focal_weight or adjusting focal_gamma")
    
    print("="*100)


def save_test_results(dataset_metrics, overall_metrics, stats_dict, test_info, args):
    """Lưu kết quả test vào file để tích lũy theo thời gian"""
    import json
    import pandas as pd
    from datetime import datetime
    
    # Tạo thư mục results nếu chưa có
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Thông tin test session
    test_session = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': test_info['model_name'],
        'backbone': test_info['backbone'],
        'loss_type': getattr(args, 'loss_type', 'structure'),
        'epochs': getattr(args, 'epochs', 'unknown'),
        'batch_size': getattr(args, 'batch_size', getattr(args, 'batchsize', 'unknown')),  # Try both batch_size and batchsize
        'weight_file': args.weight if args.weight else 'random_weights',
        'total_images': test_info['total_images'],
        'total_params': test_info['total_params'],
        'test_time': test_info['test_time'],
        'overall_metrics': {
            'dice': overall_metrics[1],
            'miou': overall_metrics[0], 
            'precision': overall_metrics[2],
            'recall': overall_metrics[3]
        },
        'dataset_metrics': dataset_metrics
    }
    
    # Lưu vào JSON (detailed results)
    json_file = os.path.join(results_dir, 'test_results_detailed.json')
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(test_session)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Lưu vào CSV (summary table)
    csv_file = os.path.join(results_dir, 'test_results_summary.csv')
    
    # Chuẩn bị dữ liệu cho CSV
    rows = []
    for dataset, metrics in dataset_metrics.items():
        row = {
            'Timestamp': test_session['timestamp'],
            'Model': test_session['model_name'],
            'Backbone': test_session['backbone'],
            'Loss_Type': test_session['loss_type'],
            'Epochs': test_session['epochs'],
            'Batch_Size': test_session['batch_size'],
            'Dataset': dataset,
            'Images': metrics['images'],
            'Dice': metrics['dice'],
            'mIoU': metrics['miou'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1_score'],
            'Total_Params': test_session['total_params'],
            'Test_Time': test_session['test_time']
        }
        rows.append(row)
    
    # Overall row
    overall_row = {
        'Timestamp': test_session['timestamp'],
        'Model': test_session['model_name'],
        'Backbone': test_session['backbone'],
        'Loss_Type': test_session['loss_type'],
        'Epochs': test_session['epochs'],
        'Batch_Size': test_session['batch_size'],
        'Dataset': 'OVERALL',
        'Images': test_session['total_images'],
        'Dice': test_session['overall_metrics']['dice'],
        'mIoU': test_session['overall_metrics']['miou'],
        'Precision': test_session['overall_metrics']['precision'],
        'Recall': test_session['overall_metrics']['recall'],
        'Accuracy': stats_dict['accuracy']['mean'],
        'F1_Score': stats_dict['f1_score']['mean'],
        'Total_Params': test_session['total_params'],
        'Test_Time': test_session['test_time']
    }
    rows.append(overall_row)
    
    # Append to CSV
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_file, index=False)
    
    print(f"\n✅ Results saved to:")
    print(f"   📁 {json_file}")
    print(f"   📊 {csv_file}")
    
    return json_file, csv_file


def display_all_test_results():
    """Hiển thị tất cả kết quả test đã lưu"""
    results_dir = 'test_results'
    csv_file = os.path.join(results_dir, 'test_results_summary.csv')
    
    if not os.path.exists(csv_file):
        print("❌ No test results found. Run some tests first!")
        return
    
    df = pd.read_csv(csv_file)
    
    # Group by model and display
    print("\n" + "="*120)
    print(f"{'COMPLETE TEST RESULTS SUMMARY':^120}")
    print("="*120)
    
    models = df['Model'].unique()
    
    for model in models:
        model_df = df[df['Model'] == model]
        
        # Get model info from most recent test
        latest = model_df.iloc[-1]
        
        print(f"\n🔹 {model} (Backbone: {latest['Backbone']}, Loss: {latest['Loss_Type']}, Epochs: {latest['Epochs']})")
        print(f"   Parameters: {latest['Total_Params']:,} | Last Test: {latest['Timestamp']}")
        print("-" * 120)
        print(f"{'Dataset':<18} | {'Images':<6} | {'Dice':<6} | {'mIoU':<6} | {'Precision':<9} | {'Recall':<6} | {'Accuracy':<8} | {'F1':<6}")
        print("-" * 120)
        
        # Get latest results for each dataset
        latest_results = model_df.groupby('Dataset').last().reset_index()
        
        # Sort by performance (Dice score), but put OVERALL at the end
        dataset_results = latest_results[latest_results['Dataset'] != 'OVERALL'].sort_values('Dice', ascending=False)
        overall_results = latest_results[latest_results['Dataset'] == 'OVERALL']
        
        for _, row in dataset_results.iterrows():
            print(f"{row['Dataset']:<18} | {int(row['Images']):<6} | {row['Dice']:<6.3f} | {row['mIoU']:<6.3f} | "
                  f"{row['Precision']:<9.3f} | {row['Recall']:<6.3f} | {row['Accuracy']:<8.3f} | {row['F1_Score']:<6.3f}")
        
        if not overall_results.empty:
            print("-" * 120)
            row = overall_results.iloc[0]
            print(f"{row['Dataset']:<18} | {int(row['Images']):<6} | {row['Dice']:<6.3f} | {row['mIoU']:<6.3f} | "
                  f"{row['Precision']:<9.3f} | {row['Recall']:<6.3f} | {row['Accuracy']:<8.3f} | {row['F1_Score']:<6.3f}")
        
        print("=" * 120)


def compare_models():
    """So sánh performance giữa các models"""
    results_dir = 'test_results'
    csv_file = os.path.join(results_dir, 'test_results_summary.csv')
    
    if not os.path.exists(csv_file):
        print("❌ No test results found for comparison!")
        return
    
    df = pd.read_csv(csv_file)
    
    # Get overall results for each model
    overall_df = df[df['Dataset'] == 'OVERALL'].groupby('Model').last().reset_index()
    
    if len(overall_df) < 2:
        print("❌ Need at least 2 models for comparison!")
        return
    
    print(f"\n{'MODEL COMPARISON':^100}")
    print("="*100)
    print(f"{'Model':<20} | {'Backbone':<8} | {'Loss':<12} | {'Dice':<6} | {'mIoU':<6} | {'Params':<10} | {'Rank'}")
    print("-"*100)
    
    # Sort by Dice score
    overall_df = overall_df.sort_values('Dice', ascending=False)
    
    for i, (_, row) in enumerate(overall_df.iterrows(), 1):
        print(f"{row['Model']:<20} | {row['Backbone']:<8} | {row['Loss_Type']:<12} | "
              f"{row['Dice']:<6.3f} | {row['mIoU']:<6.3f} | {row['Total_Params']:<10,} | #{i}")
    
    print("="*100)


def create_model_identifier(metadata):
    """Tạo unique identifier cho model từ metadata"""
    if not metadata:
        return None
    
    # Extract key information
    model_name = metadata.get('model_name', 'Unknown')
    backbone = metadata.get('backbone', 'unknown')
    epochs = metadata.get('current_epoch', metadata.get('num_epochs', 'unknown'))
    batch_size = metadata.get('batch_size', 'unknown')
    loss_type = metadata.get('loss_type', 'unknown')
    
    # Create base identifier
    identifier = f"{model_name}_{backbone}_{epochs}ep_{batch_size}bs_{loss_type}"
    
    # Add loss-specific parameters
    loss_params = metadata.get('loss_params', {})
    if loss_type == 'tversky' and loss_params:
        alpha = loss_params.get('tversky_alpha', 0.7)
        beta = loss_params.get('tversky_beta', 0.3)
        identifier += f"_a{alpha}_b{beta}"
    elif loss_type == 'combo' and loss_params:
        alpha = loss_params.get('combo_alpha', 0.5)
        identifier += f"_a{alpha}"
    elif loss_type == 'focal' and loss_params:
        gamma = loss_params.get('focal_gamma', 2.0)
        identifier += f"_g{gamma}"
    elif loss_type == 'unified_focal' and loss_params:
        gamma = loss_params.get('focal_gamma', 2.0)
        delta = loss_params.get('focal_delta', 0.6)
        identifier += f"_g{gamma}_d{delta}"
    
    return identifier


def extract_metadata_from_filename(weight_path):
    """Extract metadata từ tên file/folder với debug messages"""
    folder_name = os.path.basename(os.path.dirname(weight_path))
    
    print(f"   🔍 Parsing folder name: {folder_name}")
    
    # Try to parse folder name format: ModelName_backbone_epochs_batchsize_loss_timestamp_params
    parts = folder_name.split('_')
    print(f"   📝 Split parts: {parts}")
    
    if len(parts) >= 5:
        try:
            # Extract model name (might contain hyphens)
            model_parts = []
            i = 0
            while i < len(parts) and not parts[i].startswith(('b1', 'b2', 'b3', 'b4', 'b5')):
                model_parts.append(parts[i])
                i += 1
                
            if i < len(parts):
                model_name = '_'.join(model_parts)
                backbone = parts[i]
                
                print(f"   📋 Found model_name: {model_name}, backbone: {backbone}")
                
                # Find epochs (contains 'ep')
                epochs = 'unknown'
                batch_size = 'unknown' 
                loss_type = 'unknown'
                loss_params = {}
                
                # Scan through remaining parts to find components
                for j in range(i+1, len(parts)):
                    part = parts[j]
                    
                    if 'ep' in part:
                        epochs = part.replace('ep', '')
                        print(f"   📊 Found epochs: {epochs}")
                    elif 'bs' in part:
                        batch_size = part.replace('bs', '')
                        print(f"   📦 Found batch_size: {batch_size}")
                    elif part in ['structure', 'dice', 'tversky', 'combo', 'focal', 'unified_focal', 'boundary']:
                        loss_type = part
                        print(f"   🎯 Found loss_type: {loss_type}")
                        
                        # Look for loss-specific parameters in remaining parts
                        if loss_type == 'tversky':
                            # Look for alpha and beta parameters (format: a0.7, b0.3)
                            for k in range(j+1, len(parts)):
                                if parts[k].startswith('a') and len(parts[k]) > 1:
                                    try:
                                        loss_params['tversky_alpha'] = float(parts[k][1:])
                                        print(f"   🔧 Found tversky_alpha: {loss_params['tversky_alpha']}")
                                    except ValueError:
                                        pass
                                elif parts[k].startswith('b') and len(parts[k]) > 1:
                                    try:
                                        loss_params['tversky_beta'] = float(parts[k][1:])
                                        print(f"   🔧 Found tversky_beta: {loss_params['tversky_beta']}")
                                    except ValueError:
                                        pass
                        
                        elif loss_type == 'combo':
                            # Look for alpha parameter (format: a0.5)
                            for k in range(j+1, len(parts)):
                                if parts[k].startswith('a') and len(parts[k]) > 1:
                                    try:
                                        loss_params['combo_alpha'] = float(parts[k][1:])
                                        print(f"   🔧 Found combo_alpha: {loss_params['combo_alpha']}")
                                    except ValueError:
                                        pass
                        
                        elif loss_type == 'focal':
                            # Look for gamma parameter (format: g2.0)
                            for k in range(j+1, len(parts)):
                                if parts[k].startswith('g') and len(parts[k]) > 1:
                                    try:
                                        loss_params['focal_gamma'] = float(parts[k][1:])
                                        print(f"   🔧 Found focal_gamma: {loss_params['focal_gamma']}")
                                    except ValueError:
                                        pass
                        
                        elif loss_type == 'unified_focal':
                            # Look for gamma and delta parameters (format: g2.0, d0.6)
                            for k in range(j+1, len(parts)):
                                if parts[k].startswith('g') and len(parts[k]) > 1:
                                    try:
                                        loss_params['focal_gamma'] = float(parts[k][1:])
                                        print(f"   🔧 Found focal_gamma: {loss_params['focal_gamma']}")
                                    except ValueError:
                                        pass
                                elif parts[k].startswith('d') and len(parts[k]) > 1:
                                    try:
                                        loss_params['focal_delta'] = float(parts[k][1:])
                                        print(f"   🔧 Found focal_delta: {loss_params['focal_delta']}")
                                    except ValueError:
                                        pass
                        
                        break  # Found loss type, stop looking
                
                result_metadata = {
                    'model_name': model_name,
                    'backbone': backbone,
                    'current_epoch': epochs,
                    'num_epochs': epochs,
                    'batch_size': batch_size,
                    'loss_type': loss_type,
                    'loss_params': loss_params,
                    'extracted_from': 'filename'
                }
                
                print(f"   ✅ Extracted metadata: {result_metadata}")
                return result_metadata
                
        except Exception as e:
            print(f"   ⚠️  Error parsing filename: {e}")
            pass
    
    # Default fallback
    fallback_metadata = {
        'model_name': 'Unknown',
        'backbone': 'unknown',
        'current_epoch': 'unknown',
        'num_epochs': 'unknown', 
        'batch_size': 'unknown',
        'loss_type': 'unknown',
        'loss_params': {},
        'extracted_from': 'fallback'
    }
    
    print(f"   ❌ Using fallback metadata: {fallback_metadata}")
    return fallback_metadata


def extract_metadata_from_checkpoint(weight_path):
    """Extract metadata từ checkpoint file, ưu tiên thông tin từ folder name"""
    try:
        # First try to extract from folder name (more reliable) - silent version
        folder_name = os.path.basename(os.path.dirname(weight_path))
        parts = folder_name.split('_')
        
        folder_metadata = None
        if len(parts) >= 5:
            try:
                # Extract model name (might contain hyphens)
                model_parts = []
                i = 0
                while i < len(parts) and not parts[i].startswith(('b1', 'b2', 'b3', 'b4', 'b5')):
                    model_parts.append(parts[i])
                    i += 1
                    
                if i < len(parts):
                    model_name = '_'.join(model_parts)
                    backbone = parts[i]
                    
                    # Find epochs, batch_size, loss_type
                    epochs = 'unknown'
                    batch_size = 'unknown' 
                    loss_type = 'unknown'
                    loss_params = {}
                    
                    for j in range(i+1, len(parts)):
                        part = parts[j]
                        
                        if 'ep' in part:
                            epochs = part.replace('ep', '')
                        elif 'bs' in part:
                            batch_size = part.replace('bs', '')
                        elif part in ['structure', 'dice', 'tversky', 'combo', 'focal', 'unified_focal', 'boundary']:
                            loss_type = part
                            
                            # Look for loss-specific parameters
                            if loss_type == 'tversky':
                                for k in range(j+1, len(parts)):
                                    if parts[k].startswith('a') and len(parts[k]) > 1:
                                        try:
                                            loss_params['tversky_alpha'] = float(parts[k][1:])
                                        except ValueError:
                                            pass
                                    elif parts[k].startswith('b') and len(parts[k]) > 1:
                                        try:
                                            loss_params['tversky_beta'] = float(parts[k][1:])
                                        except ValueError:
                                            pass
                            
                            elif loss_type == 'combo':
                                for k in range(j+1, len(parts)):
                                    if parts[k].startswith('a') and len(parts[k]) > 1:
                                        try:
                                            loss_params['combo_alpha'] = float(parts[k][1:])
                                        except ValueError:
                                            pass
                            
                            elif loss_type == 'focal':
                                for k in range(j+1, len(parts)):
                                    if parts[k].startswith('g') and len(parts[k]) > 1:
                                        try:
                                            loss_params['focal_gamma'] = float(parts[k][1:])
                                        except ValueError:
                                            pass
                            
                            elif loss_type == 'unified_focal':
                                for k in range(j+1, len(parts)):
                                    if parts[k].startswith('g') and len(parts[k]) > 1:
                                        try:
                                            loss_params['focal_gamma'] = float(parts[k][1:])
                                        except ValueError:
                                            pass
                                    elif parts[k].startswith('d') and len(parts[k]) > 1:
                                        try:
                                            loss_params['focal_delta'] = float(parts[k][1:])
                                        except ValueError:
                                            pass
                            
                            break
                    
                    folder_metadata = {
                        'model_name': model_name,
                        'backbone': backbone,
                        'current_epoch': epochs,
                        'num_epochs': epochs,
                        'batch_size': batch_size,
                        'loss_type': loss_type,
                        'loss_params': loss_params,
                        'extracted_from': 'filename'
                    }
            except:
                pass
        
        # If folder metadata extraction was successful (not fallback), use it
        if folder_metadata and folder_metadata.get('loss_type') != 'unknown':
            print(f"   ✅ Using folder metadata (more reliable)")
            return folder_metadata
        
        # Otherwise, try checkpoint metadata
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        if 'training_metadata' in checkpoint:
            checkpoint_metadata = checkpoint['training_metadata']
            checkpoint_metadata['extracted_from'] = 'metadata'
            print(f"   ✅ Using checkpoint metadata")
            return checkpoint_metadata
        else:
            # If no checkpoint metadata, use folder result or fallback
            if folder_metadata:
                print(f"   ⚠️  No checkpoint metadata found, using folder extraction result")
                return folder_metadata
            else:
                print(f"   ❌ Using fallback metadata")
                return {
                    'model_name': 'Unknown',
                    'backbone': 'unknown',
                    'current_epoch': 'unknown',
                    'num_epochs': 'unknown', 
                    'batch_size': 'unknown',
                    'loss_type': 'unknown',
                    'loss_params': {},
                    'extracted_from': 'fallback'
                }
            
    except Exception as e:
        print(f"   ❌ Error loading {weight_path}: {str(e)}")
        return None


def get_tested_models():
    """Lấy danh sách các model đã được test từ CSV"""
    results_dir = 'test_results'
    csv_file = os.path.join(results_dir, 'test_results_summary.csv')
    
    tested_identifiers = set()
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # Create identifier for each tested model
            for _, row in df.iterrows():
                if row['Dataset'] == 'OVERALL':  # Only count overall results
                    identifier = f"{row['Model']}_{row['Backbone']}_{row['Epochs']}ep_{row.get('Batch_Size', 'unknown')}bs_{row['Loss_Type']}"
                    tested_identifiers.add(identifier)
        except Exception as e:
            print(f"Warning: Could not read test results: {e}")
    
    return tested_identifiers


def scan_and_identify_untested_models(args):
    """Scan tất cả snapshots và identify model nào chưa được test"""
    snapshots_dir = 'snapshots'
    if not os.path.exists(snapshots_dir):
        print("❌ No snapshots directory found!")
        return []
    
    print(f"\n🔍 Scanning snapshots directory: {snapshots_dir}")
    print("="*80)
    
    # Get list of tested models
    tested_models = get_tested_models()
    print(f"📊 Found {len(tested_models)} previously tested models")
    
    # Scan all weight files
    weight_files = []
    for root, dirs, files in os.walk(snapshots_dir):
        for file in files:
            if file.endswith('.pth'):
                weight_path = os.path.join(root, file)
                weight_files.append(weight_path)
    
    if not weight_files:
        print("❌ No .pth files found in snapshots!")
        return []
    
    print(f"📁 Found {len(weight_files)} weight file(s)")
    
    untested_models = []
    tested_count = 0
    
    for weight_path in weight_files:
        print(f"\n🔍 Analyzing: {weight_path}")
        
        # Extract metadata
        metadata = extract_metadata_from_checkpoint(weight_path)
        if not metadata:
            print(f"   ⚠️  Could not extract metadata - skipping")
            continue
        
        # Create identifier
        identifier = create_model_identifier(metadata)
        if not identifier:
            print(f"   ⚠️  Could not create identifier - skipping")
            continue
        
        # Display model info
        model_name = metadata.get('model_name', 'Unknown')
        backbone = metadata.get('backbone', 'unknown')
        epochs = metadata.get('current_epoch', 'unknown')
        loss_type = metadata.get('loss_type', 'unknown')
        extraction_method = metadata.get('extracted_from', 'metadata')
        
        print(f"   📋 Model: {model_name} | Backbone: {backbone} | Epochs: {epochs} | Loss: {loss_type}")
        print(f"   🔧 Source: {extraction_method}")
        
        # Check if already tested
        if identifier in tested_models:
            print(f"   ✅ Already tested")
            tested_count += 1
        else:
            print(f"   🆕 Not tested yet")
            untested_models.append({
                'weight_path': weight_path,
                'metadata': metadata,
                'identifier': identifier
            })
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Already tested: {tested_count}")
    print(f"   🆕 Untested: {len(untested_models)}")
    print("="*80)
    
    return untested_models


def auto_test_untested_models(args):
    """Tự động test tất cả model chưa được test"""
    untested_models = scan_and_identify_untested_models(args)
    
    if not untested_models:
        print(f"\n✅ All models have been tested!")
        return
    
    print(f"\n🚀 Starting auto-testing for {len(untested_models)} untested model(s)")
    print("="*80)
    
    success_count = 0
    error_count = 0
    
    for i, model_info in enumerate(untested_models, 1):
        weight_path = model_info['weight_path']
        metadata = model_info['metadata']
        identifier = model_info['identifier']
        
        print(f"\n🧪 Testing model {i}/{len(untested_models)}")
        print(f"📁 Weight: {weight_path}")
        print(f"🔖 ID: {identifier}")
        
        # Prepare args for testing
        test_args = argparse.Namespace(
            backbone=metadata['backbone'],
            weight=weight_path,
            test_path=args.test_path,
            test_dataset=args.test_dataset,
            loss_type=metadata.get('loss_type', 'structure'),
            epochs=metadata.get('current_epoch', metadata.get('num_epochs', 'unknown')),
            batch_size=metadata.get('batch_size', 'unknown'),
            # Loss parameters
            tversky_alpha=metadata.get('loss_params', {}).get('tversky_alpha', 0.7),
            tversky_beta=metadata.get('loss_params', {}).get('tversky_beta', 0.3),
            combo_alpha=metadata.get('loss_params', {}).get('combo_alpha', 0.5),
            focal_gamma=metadata.get('loss_params', {}).get('focal_gamma', 2.0),
            focal_delta=metadata.get('loss_params', {}).get('focal_delta', 0.6),
            # Dice Weighted IoU Focal Loss parameters
            dice_weight=metadata.get('loss_params', {}).get('dice_weight', 0.4),
            iou_weight=metadata.get('loss_params', {}).get('iou_weight', 0.3),
            focal_weight=metadata.get('loss_params', {}).get('focal_weight', 0.3),
            focal_alpha=metadata.get('loss_params', {}).get('focal_alpha', 0.25),
            smooth=metadata.get('loss_params', {}).get('smooth', 1e-6),
            spatial_weight=metadata.get('loss_params', {}).get('spatial_weight', True),
        )
        
        try:
            # Load model
            model = UNet(backbone=dict(
                type='mit_{}'.format(test_args.backbone),
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
            
            # Load weights
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"   ✅ Model loaded successfully")
            
            # Run inference
            print(f"   🔄 Running inference...")
            results = inference(model, test_args)
            
            if results:
                print(f"   ✅ Testing completed!")
                print(f"      Overall Dice: {results['overall']['dice']:.4f}")
                print(f"      Overall mIoU: {results['overall']['miou']:.4f}")
                success_count += 1
            else:
                print(f"   ❌ Testing failed - no results returned")
                error_count += 1
                
        except Exception as e:
            print(f"   ❌ Error testing model: {str(e)}")
            error_count += 1
            continue
        
        print("-" * 80)
    
    print(f"\n🎉 Auto-testing completed!")
    print(f"   ✅ Successfully tested: {success_count}")
    print(f"   ❌ Failed: {error_count}")
    print(f"   📊 Run 'python test.py --show_all' to see all results")


def scan_snapshots_and_auto_test(args):
    """Wrapper function để tương thích với code cũ"""
    auto_test_untested_models(args)


def main():
    """Main function with option to display all results"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='b3', help='backbone version (b1-b5, swin_tiny, swin_small, swin_base)')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--test_path', type=str, default='./data/TestDataset', help='path to dataset')
    parser.add_argument('--test_dataset', type=str, default='all', help='specific test dataset')
    parser.add_argument('--loss_type', type=str, default='structure', help='loss type used during training')
    parser.add_argument('--tversky_alpha', type=float, default=0.7, help='alpha for Tversky Loss')
    parser.add_argument('--tversky_beta', type=float, default=0.3, help='beta for Tversky Loss')
    parser.add_argument('--combo_alpha', type=float, default=0.5, help='alpha for Combo Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for Focal Loss')
    parser.add_argument('--focal_delta', type=float, default=0.6, help='delta for Unified Focal Loss')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs used for training (for tracking)')
    
    # Dice Weighted IoU Focal Loss arguments
    parser.add_argument('--dice_weight', type=float, default=0.4, help='dice weight for dice_weighted_iou_focal loss')
    parser.add_argument('--iou_weight', type=float, default=0.3, help='iou weight for dice_weighted_iou_focal loss')
    parser.add_argument('--focal_weight', type=float, default=0.3, help='focal weight for dice_weighted_iou_focal loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='focal alpha for dice_weighted_iou_focal loss')
    parser.add_argument('--smooth', type=float, default=1e-6, help='smoothing factor for dice_weighted_iou_focal loss')
    parser.add_argument('--spatial_weight', type=bool, default=True, help='use spatial weighting for dice_weighted_iou_focal loss')
    
    # Special arguments for viewing results
    parser.add_argument('--show_all', action='store_true', help='Display all accumulated test results')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--auto_test', action='store_true', help='Auto scan snapshots and test untested models')
    
    args = parser.parse_args()

    # Special modes
    if args.show_all:
        display_all_test_results()
        return
    
    if args.compare:
        compare_models()
        return
    
    if args.auto_test:
        scan_snapshots_and_auto_test(args)
        return
    
    # Regular testing mode
    # Setup logging
    log_file = setup_logging('logs', args.test_dataset)
    logging.info(f"Log file created: {log_file}")

    logging.info("="*60)
    logging.info("COLONFORMER TESTING STARTED")
    logging.info("="*60)
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Model: {get_model_name(args.backbone)}")
    logging.info(f"Test dataset: {args.test_dataset}")
    logging.info(f"Test path: {args.test_path}")
    logging.info(f"Backbone: {args.backbone}")
    logging.info(f"Training Loss Type: {args.loss_type}")
    logging.info(
        f"Weight file: {args.weight if args.weight else 'None (Random weights)'}")

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
    else:
        # MiT backbone (mặc định)
        backbone_cfg = dict(
            type='mit_{}'.format(args.backbone),
            style='pytorch',
        )
    
    model = UNet(backbone=backbone_cfg,
        decode_head=dict(
        type='UPerHead',
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
        pretrained=None).cuda()

    # Log model information
    log_model_info(model, args)

    if args.weight != '':
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Successfully loaded weights from: {args.weight}")
        
        # Auto detect loss configuration from checkpoint metadata
        if 'training_metadata' in checkpoint:
            metadata = checkpoint['training_metadata']
            
            # Auto set loss type
            if 'loss_type' in metadata:
                detected_loss_type = metadata['loss_type']
                if args.loss_type != detected_loss_type:
                    logging.info(f"Auto detected loss type: {detected_loss_type} (overriding {args.loss_type})")
                    args.loss_type = detected_loss_type
            
            # Auto set loss parameters
            if 'loss_params' in metadata:
                loss_params = metadata['loss_params']
                logging.info("Auto detected loss parameters:")
                
                for param_name, param_value in loss_params.items():
                    if hasattr(args, param_name):
                        old_value = getattr(args, param_name)
                        if old_value != param_value:  # Only log if different
                            setattr(args, param_name, param_value)
                            logging.info(f"  {param_name}: {old_value} -> {param_value}")
                    else:
                        logging.info(f"  {param_name}: {param_value} (new parameter)")
                        setattr(args, param_name, param_value)
            
            # Log component loss info if available
            if 'component_losses' in metadata:
                comp_losses = metadata['component_losses']
                logging.info("Training Component Loss Information:")
                logging.info(f"  Training Dice Loss: {comp_losses['dice_loss']:.6f}")
                logging.info(f"  Training IoU Loss: {comp_losses['iou_loss']:.6f}")
                logging.info(f"  Training Focal Loss: {comp_losses['focal_loss']:.6f}")
                if 'weights' in comp_losses:
                    weights = comp_losses['weights']
                    logging.info(f"  Training Weights: D={weights['dice_weight']:.3f}, I={weights['iou_weight']:.3f}, F={weights['focal_weight']:.3f}")
        else:
            logging.info("No training metadata found in checkpoint, using manual parameters")

    # Run inference
    results = inference(model, args)

    logging.info("="*60)
    logging.info("TESTING COMPLETED SUCCESSFULLY")
    logging.info("="*60)
    logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if results:
        # Results are now displayed in comprehensive format above
        # Analyze performance trends if data is available
        if 'datasets' in results and 'stats' in results:
            overall_metrics = (
                results['overall']['miou'],
                results['overall']['dice'], 
                results['overall']['precision'],
                results['overall']['recall']
            )
            # Performance analysis is now included in print_comprehensive_results


if __name__ == '__main__':
    main()
