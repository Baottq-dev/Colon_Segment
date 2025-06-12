#!/usr/bin/env python3
"""
Auto detect test script cho Dice Weighted IoU Focal Loss
Tự động phát hiện loss type và parameters từ checkpoint metadata
"""

import argparse
import subprocess
import sys
import torch
import os

def detect_loss_config_from_checkpoint(checkpoint_path):
    """Phát hiện loss configuration từ checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = {
            'loss_type': 'structure',  # default
            'backbone': 'b3',  # default
        }
        
        if 'training_metadata' in checkpoint:
            metadata = checkpoint['training_metadata']
            
            # Extract basic info
            config['backbone'] = metadata.get('backbone', 'b3')
            config['loss_type'] = metadata.get('loss_type', 'structure')
            config['epochs'] = metadata.get('current_epoch', metadata.get('num_epochs', 'unknown'))
            
            # Extract loss parameters
            if 'loss_params' in metadata:
                config.update(metadata['loss_params'])
            
            # Extract component loss info if available
            if 'component_losses' in metadata:
                config['component_losses'] = metadata['component_losses']
        
        return config
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def build_test_command(checkpoint_path, test_path='./data/TestDataset'):
    """Xây dựng command test với auto detected parameters"""
    
    config = detect_loss_config_from_checkpoint(checkpoint_path)
    if not config:
        return None
    
    # Base command
    cmd = [
        'python', 'test.py',
        '--weight', checkpoint_path,
        '--test_path', test_path,
        '--backbone', config['backbone'],
        '--loss_type', config['loss_type']
    ]
    
    # Add loss-specific parameters if available
    loss_param_mapping = {
        'dice_weight': '--dice_weight',
        'iou_weight': '--iou_weight', 
        'focal_weight': '--focal_weight',
        'focal_alpha': '--focal_alpha',
        'focal_gamma': '--focal_gamma',
        'smooth': '--smooth',
        'spatial_weight': '--spatial_weight',
        'tversky_alpha': '--tversky_alpha',
        'tversky_beta': '--tversky_beta',
        'combo_alpha': '--combo_alpha',
        'focal_delta': '--focal_delta'
    }
    
    for param_name, cmd_arg in loss_param_mapping.items():
        if param_name in config:
            cmd.extend([cmd_arg, str(config[param_name])])
    
    return cmd, config

def print_detected_config(config):
    """In thông tin configuration đã detect"""
    
    print("="*80)
    print("AUTO DETECTED CONFIGURATION")
    print("="*80)
    
    print(f"Backbone: {config['backbone']}")
    print(f"Loss Type: {config['loss_type']}")
    print(f"Epochs: {config.get('epochs', 'unknown')}")
    
    if config['loss_type'] == 'dice_weighted_iou_focal':
        print(f"\nDice Weighted IoU Focal Loss Parameters:")
        print(f"  Dice Weight: {config.get('dice_weight', 0.4)}")
        print(f"  IoU Weight: {config.get('iou_weight', 0.3)}")
        print(f"  Focal Weight: {config.get('focal_weight', 0.3)}")
        print(f"  Focal Alpha: {config.get('focal_alpha', 0.25)}")
        print(f"  Focal Gamma: {config.get('focal_gamma', 2.0)}")
        print(f"  Smooth: {config.get('smooth', 1e-6)}")
        print(f"  Spatial Weight: {config.get('spatial_weight', True)}")
        
        if 'component_losses' in config:
            comp = config['component_losses']
            print(f"\nTraining Component Losses:")
            print(f"  Dice Loss: {comp['dice_loss']:.6f}")
            print(f"  IoU Loss: {comp['iou_loss']:.6f}")
            print(f"  Focal Loss: {comp['focal_loss']:.6f}")
    
    elif config['loss_type'] == 'tversky':
        print(f"\nTversky Loss Parameters:")
        print(f"  Alpha: {config.get('tversky_alpha', 0.7)}")
        print(f"  Beta: {config.get('tversky_beta', 0.3)}")
    
    elif config['loss_type'] == 'combo':
        print(f"\nCombo Loss Parameters:")
        print(f"  Alpha: {config.get('combo_alpha', 0.5)}")
    
    elif config['loss_type'] == 'unified_focal':
        print(f"\nUnified Focal Loss Parameters:")
        print(f"  Gamma: {config.get('focal_gamma', 2.0)}")
        print(f"  Delta: {config.get('focal_delta', 0.6)}")
    
    print("="*80)

def run_auto_test(checkpoint_path, test_path='./data/TestDataset', dry_run=False):
    """Chạy test với auto detected parameters"""
    
    cmd, config = build_test_command(checkpoint_path, test_path)
    if not cmd:
        return False
    
    print_detected_config(config)
    
    print(f"\nGenerated Test Command:")
    print(" ".join(cmd))
    
    if dry_run:
        print("\nDry run mode - command not executed")
        return True
    
    print(f"\nRunning test...")
    print("-"*80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-"*80)
        print("Test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test failed with return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto detect and run ColonFormer test")
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='./data/TestDataset',
                        help='Path to test dataset')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show command without executing')
    parser.add_argument('--detect_only', action='store_true',
                        help='Only detect and show configuration')
    
    args = parser.parse_args()
    
    if args.detect_only:
        config = detect_loss_config_from_checkpoint(args.checkpoint_path)
        if config:
            print_detected_config(config)
        else:
            print("Failed to detect configuration")
            sys.exit(1)
    else:
        success = run_auto_test(args.checkpoint_path, args.test_path, args.dry_run)
        if not success:
            sys.exit(1) 