#!/usr/bin/env python3
"""
Complete Auto System Demo for ColonFormer
Hệ thống tự động hoàn toàn: từ training, auto-naming, auto-rename, auto-test đến result tracking
"""

import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


class ColonFormerAutoSystem:
    def __init__(self):
        self.snapshots_dir = Path("snapshots")
        self.results_dir = Path("test_results")
        
    def show_status(self):
        """Hiển thị trạng thái hệ thống"""
        print("\n" + "="*80)
        print("🔍 COLONFORMER AUTO SYSTEM STATUS")
        print("="*80)
        
        # Snapshots status
        if self.snapshots_dir.exists():
            model_files = list(self.snapshots_dir.rglob("*.pth"))
            print(f"📁 Snapshots: {len(model_files)} model file(s)")
            
            # Analyze naming convention
            good_names = 0
            old_names = 0
            
            for model_file in model_files:
                folder_name = model_file.parent.name
                if self._is_good_naming(folder_name):
                    good_names += 1
                else:
                    old_names += 1
            
            print(f"   ✅ Good naming: {good_names}")
            print(f"   ⚠️  Old naming: {old_names}")
        else:
            print("📁 Snapshots: Directory not found")
        
        # Test results status
        if self.results_dir.exists():
            csv_file = self.results_dir / "test_results_summary.csv"
            if csv_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    unique_models = df['Model'].unique()
                    print(f"📊 Test Results: {len(df)} records, {len(unique_models)} unique models")
                except:
                    print("📊 Test Results: Found but cannot read")
            else:
                print("📊 Test Results: No results yet")
        else:
            print("📊 Test Results: Directory not found")
    
    def _is_good_naming(self, folder_name):
        """Check if folder follows good naming convention"""
        import re
        patterns = [
            r'^ColonFormer[A-Z]{1,3}_b[0-5]_\d+ep_\d+bs_\w+',  # Full convention
            r'^ColonFormer[A-Z]{1,3}_b[0-5]_\d+ep_\w+',        # Without batch size
        ]
        
        for pattern in patterns:
            if re.match(pattern, folder_name):
                return True
        return False
    
    def auto_rename_snapshots(self, execute=False):
        """Tự động đổi tên snapshots theo convention mới"""
        print("\n" + "="*80)
        print("🏷️  AUTO-RENAME SNAPSHOTS")
        print("="*80)
        
        cmd = ['python', 'auto_rename_snapshots.py', '--skip_empty']
        if execute:
            cmd.append('--execute')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings:")
            print(result.stderr)
    
    def auto_test_models(self, test_path="./data/TestDataset"):
        """Tự động test các model chưa được test"""
        print("\n" + "="*80) 
        print("🧪 AUTO-TEST UNTESTED MODELS")
        print("="*80)
        
        cmd = ['python', 'test.py', '--auto_test', '--test_path', test_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings:")
            print(result.stderr)
    
    def show_all_results(self):
        """Hiển thị tất cả kết quả test"""
        print("\n" + "="*80)
        print("📊 ALL TEST RESULTS")
        print("="*80)
        
        cmd = ['python', 'test.py', '--show_all']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
    
    def compare_models(self):
        """So sánh performance giữa các models"""
        print("\n" + "="*80)
        print("🏆 MODEL COMPARISON")
        print("="*80)
        
        cmd = ['python', 'test.py', '--compare']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
    
    def demo_training(self, execute=False):
        """Demo training với auto-naming"""
        print("\n" + "="*80)
        print("🚀 DEMO: AUTO-NAMING TRAINING")
        print("="*80)
        
        demo_configs = [
            {
                'backbone': 'b2',
                'epochs': 3,
                'batch_size': 16,
                'loss_type': 'dice'
            },
            {
                'backbone': 'b3', 
                'epochs': 5,
                'batch_size': 32,
                'loss_type': 'tversky',
                'tversky_alpha': 0.8,
                'tversky_beta': 0.2
            }
        ]
        
        print(f"📋 Demo training configs:")
        for i, config in enumerate(demo_configs, 1):
            print(f"   {i}. ColonFormer-{config['backbone'].upper()}: {config['epochs']}ep, {config['batch_size']}bs, {config['loss_type']}")
        
        if execute:
            print("\n🏃‍♂️ Executing demo training...")
            for config in demo_configs:
                cmd = [
                    'python', 'train.py',
                    '--backbone', config['backbone'],
                    '--num_epochs', str(config['epochs']),
                    '--batchsize', str(config['batch_size']),
                    '--loss_type', config['loss_type']
                ]
                
                if 'tversky_alpha' in config:
                    cmd.extend(['--tversky_alpha', str(config['tversky_alpha'])])
                    cmd.extend(['--tversky_beta', str(config['tversky_beta'])])
                
                print(f"\n🚀 Starting training: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("   ✅ Training completed")
                else:
                    print("   ❌ Training failed")
                    print(result.stderr[:500])  # Show first 500 chars of error
        else:
            print("\n💡 Use --execute to run actual training")
    
    def complete_workflow(self, execute_training=False, execute_rename=False):
        """Complete workflow từ đầu đến cuối"""
        print("🔄 COMPLETE AUTO WORKFLOW")
        print("="*80)
        
        # Step 1: Show current status
        self.show_status()
        
        # Step 2: Demo training (optional)
        self.demo_training(execute=execute_training)
        
        # Step 3: Auto-rename snapshots
        self.auto_rename_snapshots(execute=execute_rename)
        
        # Step 4: Auto-test models
        self.auto_test_models()
        
        # Step 5: Show results
        self.show_all_results()
        
        # Step 6: Compare models
        self.compare_models()
        
        print("\n✅ Complete workflow finished!")
    
    def maintenance_mode(self):
        """Maintenance tasks"""
        print("\n" + "="*80)
        print("🔧 MAINTENANCE MODE")
        print("="*80)
        
        # Clean up empty folders
        print("🧹 Cleaning up empty folders...")
        empty_count = 0
        
        if self.snapshots_dir.exists():
            for folder in self.snapshots_dir.iterdir():
                if folder.is_dir():
                    if not any(folder.iterdir()):  # Empty folder
                        print(f"   📁 Empty folder: {folder.name}")
                        empty_count += 1
        
        print(f"   Found {empty_count} empty folders")
        
        # Check naming consistency
        print("\n🏷️  Checking naming consistency...")
        inconsistent_count = 0
        
        if self.snapshots_dir.exists():
            for folder in self.snapshots_dir.iterdir():
                if folder.is_dir() and not self._is_good_naming(folder.name):
                    print(f"   ⚠️  Inconsistent naming: {folder.name}")
                    inconsistent_count += 1
        
        print(f"   Found {inconsistent_count} folders with inconsistent naming")
        
        # Check for missing test results
        print("\n📊 Checking test coverage...")
        
        cmd = ['python', 'test.py', '--auto_test']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "untested" in result.stdout.lower():
            print("   ⚠️  Found untested models")
        else:
            print("   ✅ All models tested")


def main():
    parser = argparse.ArgumentParser(description="ColonFormer Complete Auto System")
    parser.add_argument('--mode', choices=['status', 'rename', 'test', 'results', 'compare', 'training', 'complete', 'maintenance'], 
                        default='status', help='Operation mode')
    parser.add_argument('--execute', action='store_true', help='Execute operations (not just preview)')
    parser.add_argument('--test_path', default='./data/TestDataset', help='Path to test dataset')
    
    args = parser.parse_args()
    
    system = ColonFormerAutoSystem()
    
    print("🤖 COLONFORMER COMPLETE AUTO SYSTEM")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.mode == 'status':
            system.show_status()
        elif args.mode == 'rename':
            system.auto_rename_snapshots(execute=args.execute)
        elif args.mode == 'test':
            system.auto_test_models(test_path=args.test_path)
        elif args.mode == 'results':
            system.show_all_results()
        elif args.mode == 'compare':
            system.compare_models()
        elif args.mode == 'training':
            system.demo_training(execute=args.execute)
        elif args.mode == 'complete':
            system.complete_workflow(execute_training=args.execute, execute_rename=args.execute)
        elif args.mode == 'maintenance':
            system.maintenance_mode()
        
        print(f"\n🎉 Operation '{args.mode}' completed successfully!")
        
        print("\n📚 Available commands:")
        print("  python complete_auto_system.py --mode status        # Show system status")
        print("  python complete_auto_system.py --mode rename        # Auto-rename snapshots")
        print("  python complete_auto_system.py --mode test          # Auto-test models")
        print("  python complete_auto_system.py --mode results       # Show all results")
        print("  python complete_auto_system.py --mode compare       # Compare models")
        print("  python complete_auto_system.py --mode training      # Demo training")
        print("  python complete_auto_system.py --mode complete      # Complete workflow")
        print("  python complete_auto_system.py --mode maintenance   # System maintenance")
        print("\n  Add --execute to run actual operations (not just preview)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 