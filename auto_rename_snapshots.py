#!/usr/bin/env python3
"""
Auto Rename Snapshots Script for ColonFormer
Tự động detect và đổi tên các thư mục snapshots theo convention mới
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class AutoRenameSnapshots:
    def __init__(self, snapshots_dir: str = "snapshots"):
        self.snapshots_dir = Path(snapshots_dir)
        
        # Mapping backbone names
        self.backbone_mapping = {
            'b0': 'XS',
            'b1': 'XS', 
            'b2': 'S',
            'b3': 'L',
            'b4': 'XL',
            'b5': 'XXL'
        }
        
        # Pattern để detect folder names cũ
        self.old_patterns = [
            r'ColonFormer[Bb]([0-5])',  # ColonFormerB3, ColonFormerb3
            r'ColonFormer([XSLM]+)',    # ColonFormerL, ColonFormerXS, etc.
            r'MiT[_-]?[Bb]([0-5])',     # MiT-B3, MiT_B3, MiTB3
            r'mit[_-]?[Bb]([0-5])',     # mit-b3, mit_b3
            r'segformer[_-]?[Bb]([0-5])', # segformer-b3
        ]
        
        # Reverse mapping để detect size names đã có
        self.size_to_backbone = {
            'XS': 'b1',
            'S': 'b2', 
            'L': 'b3',
            'XL': 'b4',
            'XXL': 'b5'
        }
        
    def parse_folder_name(self, folder_name: str) -> Optional[Dict]:
        """
        Parse thông tin từ tên folder để extract backbone, epochs, loss type, etc.
        """
        info = {
            'original_name': folder_name,
            'backbone_old': None,
            'backbone_new': None,
            'epochs': None,
            'loss_type': None,
            'additional_info': []
        }
        
        # Detect backbone
        for pattern in self.old_patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                backbone_identifier = match.group(1)
                
                # Check if it's a numeric backbone (b0-b5)
                if backbone_identifier.isdigit():
                    backbone_num = backbone_identifier
                    info['backbone_old'] = f'b{backbone_num}'
                    info['backbone_new'] = self.backbone_mapping.get(f'b{backbone_num}', f'B{backbone_num}')
                else:
                    # It's a size name (XS, S, L, XL, XXL)
                    size_name = backbone_identifier.upper()
                    if size_name in self.size_to_backbone:
                        info['backbone_old'] = self.size_to_backbone[size_name]
                        info['backbone_new'] = size_name
                    else:
                        # Handle partial matches like 'L' in 'XL'
                        for size, backbone in self.size_to_backbone.items():
                            if size in size_name:
                                info['backbone_old'] = backbone
                                info['backbone_new'] = size
                                break
                break
        
        if not info['backbone_old']:
            return None
            
        # Extract epochs
        epoch_patterns = [
            r'(\d+)ep(?:och)?s?',
            r'ep(?:och)?s?(\d+)',
            r'(\d+)_?ep',
            r'epoch_?(\d+)'
        ]
        
        for pattern in epoch_patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                info['epochs'] = match.group(1)
                break
                
        # Extract loss type
        loss_patterns = [
            r'(dice)',
            r'(focal)',
            r'(cross_?entropy)',
            r'(ce)',
            r'(structure)',
            r'(boundary)'
        ]
        
        for pattern in loss_patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                info['loss_type'] = match.group(1).lower()
                break
                
        # Extract additional info
        additional_keywords = [
            'pretrained', 'finetune', 'scratch', 'augment', 'dropout',
            'lr', 'batch', 'size', 'adam', 'sgd', 'warmup', 'cosine'
        ]
        
        for keyword in additional_keywords:
            if re.search(rf'\b{keyword}\b', folder_name, re.IGNORECASE):
                info['additional_info'].append(keyword)
                
        return info
        
    def generate_new_name(self, info: Dict, add_timestamp: bool = True) -> str:
        """
        Generate tên mới theo convention: ColonFormer{Size}_{backbone}_{epochs}ep_{batchsize}bs_{loss_type}_TIMESTAMP
        """
        parts = ['ColonFormer' + info['backbone_new']]
        
        if info['backbone_old']:
            parts.append(info['backbone_old'])
            
        if info['epochs']:
            parts.append(f"{info['epochs']}ep")
        else:
            # Default epochs if not found
            parts.append("20ep")
            
        # Try to detect batch size from folder name
        batch_size = self._extract_batch_size(info['original_name'])
        if batch_size:
            parts.append(f"{batch_size}bs")
        else:
            parts.append("32bs")  # Default batch size
            
        if info['loss_type']:
            parts.append(info['loss_type'])
        else:
            parts.append("structure")  # Default loss type
            
        # Handle timestamp
        existing_timestamp = self._extract_timestamp(info['original_name'])
        
        if add_timestamp:
            if existing_timestamp:
                parts.append(existing_timestamp)
            else:
                # If no timestamp found, suggest adding one
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                parts.append(timestamp)
        elif existing_timestamp:
            # Keep existing timestamp even if add_timestamp=False
            parts.append(existing_timestamp)
            
        if info['additional_info']:
            # Insert additional info before timestamp
            try:
                if len(parts) > 0 and parts[-1] and re.match(r'\d{8}_\d{4}|\d{8}', parts[-1]):  # If last part looks like timestamp
                    timestamp = parts.pop()  # Remove timestamp
                    parts.extend(info['additional_info'])
                    parts.append(timestamp)  # Add timestamp back
                else:
                    parts.extend(info['additional_info'])
            except Exception as e:
                print(f"   ⚠️  Error handling additional info: {e}")
                parts.extend(info['additional_info'])
            
        return '_'.join(parts)
        
    def _extract_batch_size(self, folder_name: str) -> Optional[str]:
        """Extract batch size từ folder name"""
        batch_patterns = [
            r'(\d+)bs\b',              # 32bs (word boundary)
            r'batch_?size_?(\d+)',     # batch_size_32, batchsize32
            r'batch_?(\d+)',           # batch32, batch_32
            r'\bbs(\d+)',              # bs32 (word boundary)
        ]
        
        for pattern in batch_patterns:
            match = re.search(pattern, folder_name, re.IGNORECASE)
            if match:
                batch_size = int(match.group(1))
                # Reasonable batch size range (4-128)
                if 4 <= batch_size <= 128:
                    return str(batch_size)
        return None
        
    def _extract_timestamp(self, folder_name: str) -> Optional[str]:
        """Extract timestamp từ folder name"""
        timestamp_patterns = [
            r'(\d{8}_\d{4})',          # 20241201_1430
            r'(\d{4}\d{2}\d{2}_\d{4})', # 20241201_1430
            r'(\d{6}_\d{4})',          # 241201_1430  
            r'(\d{8})',                # 20241201
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, folder_name)
            if match:
                return match.group(1)
        return None
        
    def check_folder_structure(self, folder_path: Path) -> Dict:
        """
        Kiểm tra cấu trúc bên trong folder để xác nhận đây là valid snapshot
        """
        structure_info = {
            'is_valid_snapshot': False,
            'has_checkpoints': False,
            'checkpoint_files': [],
            'has_config': False,
            'config_files': []
        }
        
        if not folder_path.exists() or not folder_path.is_dir():
            return structure_info
            
        # Check for checkpoint files
        checkpoint_extensions = ['.pth', '.pt', '.ckpt', '.pkl']
        for ext in checkpoint_extensions:
            checkpoints = list(folder_path.glob(f'*{ext}'))
            if checkpoints:
                structure_info['has_checkpoints'] = True
                structure_info['checkpoint_files'].extend([f.name for f in checkpoints])
                
        # Check for config files
        config_patterns = ['*.json', '*.yaml', '*.yml', '*config*']
        for pattern in config_patterns:
            configs = list(folder_path.glob(pattern))
            if configs:
                structure_info['has_config'] = True
                structure_info['config_files'].extend([f.name for f in configs])
                
        structure_info['is_valid_snapshot'] = (
            structure_info['has_checkpoints'] or 
            structure_info['has_config'] or
            len(list(folder_path.iterdir())) > 0
        )
        
        return structure_info
        
    def scan_snapshots(self) -> List[Dict]:
        """
        Scan tất cả folders trong snapshots directory
        """
        if not self.snapshots_dir.exists():
            print(f"❌ Thư mục snapshots không tồn tại: {self.snapshots_dir}")
            return []
            
        scan_results = []
        
        for folder in self.snapshots_dir.iterdir():
            if not folder.is_dir():
                continue
                
            print(f"🔍 Scanning folder: {folder.name}")
                
            # Parse folder name
            info = self.parse_folder_name(folder.name)
            if not info:
                print(f"   ⚠️  Cannot parse folder name: {folder.name}")
                continue
                
            print(f"   ✅ Parsed: {info['backbone_old']} → {info['backbone_new']}")
                
            # Check folder structure
            structure = self.check_folder_structure(folder)
            
            # Check if name is already good
            is_already_good = self.is_good_name(folder.name)
            
            # Generate new name (without timestamp if already good)
            new_name = self.generate_new_name(info, add_timestamp=not is_already_good)
            
            result = {
                'current_path': folder,
                'current_name': folder.name,
                'suggested_name': new_name,
                'needs_rename': folder.name != new_name,
                'is_good_name': is_already_good,
                'info': info,
                'structure': structure
            }
            
            scan_results.append(result)
            
        return scan_results
        
    def display_scan_results(self, results: List[Dict]):
        """
        Hiển thị kết quả scan dưới dạng table
        """
        if not results:
            print("📝 Không tìm thấy snapshots nào cần đổi tên")
            return
            
        print("\n" + "="*80)
        print("🔍 KẾT QUẢ SCAN SNAPSHOTS")
        print("="*80)
        
        needs_rename = [r for r in results if r['needs_rename']]
        already_renamed = [r for r in results if not r['needs_rename']]
        
        if needs_rename:
            print(f"\n📋 CÁC FOLDER CẦN ĐỔI TÊN ({len(needs_rename)} folders):")
            print("-" * 80)
            
            for i, result in enumerate(needs_rename, 1):
                info = result['info']
                structure = result['structure']
                
                print(f"\n{i}. 📁 {result['current_name']}")
                print(f"   ➜ {result['suggested_name']}")
                print(f"   📊 Backbone: {info['backbone_old']} → ColonFormer-{info['backbone_new']}")
                
                if info['epochs']:
                    print(f"   ⏱️  Epochs: {info['epochs']}")
                if info['loss_type']:
                    print(f"   🎯 Loss: {info['loss_type']}")
                if info['additional_info']:
                    print(f"   ℹ️  Extra: {', '.join(info['additional_info'])}")
                    
                if structure['is_valid_snapshot']:
                    print(f"   ✅ Valid snapshot")
                    if structure['checkpoint_files']:
                        print(f"   📦 Checkpoints: {len(structure['checkpoint_files'])} files")
                else:
                    print(f"   ⚠️  Không phải valid snapshot")
                    
        if already_renamed:
            print(f"\n✅ CÁC FOLDER ĐÃ ĐÚNG TÊN ({len(already_renamed)} folders):")
            print("-" * 80)
            for result in already_renamed:
                print(f"   📁 {result['current_name']}")
                
        print(f"\n📊 TỔNG KẾT:")
        print(f"   • Tổng số folders: {len(results)}")
        print(f"   • Cần đổi tên: {len(needs_rename)}")
        print(f"   • Đã đúng tên: {len(already_renamed)}")
        
    def perform_rename(self, results: List[Dict], dry_run: bool = True):
        """
        Thực hiện đổi tên các folders
        """
        needs_rename = [r for r in results if r['needs_rename']]
        
        if not needs_rename:
            print("✅ Tất cả folders đã có tên đúng!")
            return
            
        if dry_run:
            print(f"\n🔍 DRY RUN - Sẽ thực hiện {len(needs_rename)} operations:")
        else:
            print(f"\n🚀 THỰC HIỆN ĐỔI TÊN {len(needs_rename)} folders:")
            
        success_count = 0
        error_count = 0
        
        for result in needs_rename:
            old_path = result['current_path']
            new_path = old_path.parent / result['suggested_name']
            
            print(f"\n📁 {result['current_name']}")
            print(f"  ➜ {result['suggested_name']}")
            
            if dry_run:
                print(f"  📝 Would rename: {old_path} → {new_path}")
                success_count += 1
            else:
                try:
                    if new_path.exists():
                        print(f"  ❌ Target already exists: {new_path}")
                        error_count += 1
                        continue
                        
                    old_path.rename(new_path)
                    print(f"  ✅ Renamed successfully!")
                    success_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    error_count += 1
                    
        print(f"\n📊 KẾT QUẢ:")
        if dry_run:
            print(f"  • Sẽ đổi tên: {success_count} folders")
        else:
            print(f"  • Thành công: {success_count} folders")
            print(f"  • Lỗi: {error_count} folders")
            
    def save_rename_log(self, results: List[Dict], log_file: str = "rename_log.json"):
        """
        Lưu log của quá trình rename
        """
        log_data = {
            'timestamp': str(Path().cwd()),
            'snapshots_dir': str(self.snapshots_dir),
            'total_folders': len(results),
            'needs_rename': len([r for r in results if r['needs_rename']]),
            'results': []
        }
        
        for result in results:
            log_entry = {
                'current_name': result['current_name'],
                'suggested_name': result['suggested_name'],
                'needs_rename': result['needs_rename'],
                'backbone_old': result['info']['backbone_old'],
                'backbone_new': result['info']['backbone_new'],
                'epochs': result['info']['epochs'],
                'loss_type': result['info']['loss_type'],
                'is_valid_snapshot': result['structure']['is_valid_snapshot']
            }
            log_data['results'].append(log_entry)
            
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Đã lưu log: {log_file}")

    def is_good_name(self, folder_name: str) -> bool:
        """Check if folder name đã theo convention tốt"""
        # Pattern for good names: ColonFormer[Size]_[backbone]_[epochs]ep_[optional]
        good_patterns = [
            r'^ColonFormer[A-Z]{1,3}_b[0-5]_\d+ep',  # ColonFormerL_b3_20ep...
            r'^ColonFormer[A-Z]{1,3}_b[0-5]_\d+ep_.*',  # ColonFormerL_b3_20ep_structure...
        ]
        
        try:
            for pattern in good_patterns:
                if re.match(pattern, folder_name):
                    return True
        except Exception as e:
            print(f"   ⚠️  Regex error in is_good_name: {e}")
            
        return False

def main():
    parser = argparse.ArgumentParser(description='Auto Rename ColonFormer Snapshots')
    parser.add_argument('--snapshots_dir', default='snapshots', 
                       help='Thư mục snapshots (default: snapshots)')
    parser.add_argument('--dry_run', action='store_true', 
                       help='Chỉ xem preview, không thực sự đổi tên')
    parser.add_argument('--execute', action='store_true',
                       help='Thực hiện đổi tên (ngược với dry_run)')
    parser.add_argument('--save_log', action='store_true',
                       help='Lưu log results vào file JSON')
    parser.add_argument('--skip_empty', action='store_true',
                       help='Bỏ qua các folder trống hoặc không có checkpoints')
    parser.add_argument('--include_all', action='store_true',
                       help='Bao gồm tất cả folders kể cả empty folders')
    
    args = parser.parse_args()
    
    try:
        # Initialize renamer
        renamer = AutoRenameSnapshots(args.snapshots_dir)
        
        print("🚀 AUTO RENAME SNAPSHOTS FOR COLONFORMER")
        print("="*50)
        
        # Scan snapshots
        print("🔍 Đang scan snapshots directory...")
        results = renamer.scan_snapshots()
        
        # Filter results based on arguments
        if args.skip_empty and not args.include_all:
            results = [r for r in results if r['structure']['is_valid_snapshot']]
            print(f"📋 Đã lọc bỏ empty folders, còn lại {len(results)} folders")
        
        # Display results
        renamer.display_scan_results(results)
        
        # Save log if requested
        if args.save_log:
            renamer.save_rename_log(results)
            
        # Perform rename
        if args.execute:
            print(f"\n⚠️  BẠN CHẮC CHẮN MUỐN ĐỔI TÊN? (y/N): ", end="")
            confirmation = input().strip().lower()
            if confirmation in ['y', 'yes', 'có']:
                renamer.perform_rename(results, dry_run=False)
            else:
                print("❌ Hủy thao tác đổi tên")
        else:
            # Default is dry run
            renamer.perform_rename(results, dry_run=True)
            print(f"\n💡 Để thực sự đổi tên, chạy với --execute")
            if not args.skip_empty:
                print(f"💡 Để bỏ qua empty folders, chạy với --skip_empty")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 