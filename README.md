# ColonFormer: An Efficient Transformer-based Method for Colon Polyp Segmentation

An implementation of ColonFormer - an efficient colon polyp segmentation method using Transformer architecture with enhanced logging, visualization, evaluation capabilities, and automated workflow system.

**Note**: This is a modified version compatible with MMCV 2.2.0. The original repository and official paper can be found at [ducnt9907/ColonFormer](https://github.com/ducnt9907/ColonFormer).

## Table of Contents
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Performance Metrics](#performance-metrics)
- [Automated Features](#automated-features)
- [Model Variants and Performance](#model-variants-and-performance)
- [Changes from Original Version](#changes-from-original-version)
- [Technical Requirements](#technical-requirements)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contributing](#contributing)
- [Issues and Support](#issues-and-support)

## Key Features

### Architecture Components

- **Mix Transformer (MiT) Backbone**: Support for MIT-B1 to MIT-B5 variants
- **Context Feature Pyramid (CFP) Module**: Multi-scale dilated convolution features (d=8)
- **Axial Attention Mechanism**: Efficient axis-wise attention computation
- **Reverse Attention**: Attention mechanism for focus refinement with three-stage processing
- **Multi-scale Output**: 4 feature maps at different resolutions (8x, 16x, 32x, 4x upsampling)

### Enhanced Features

- **Advanced Logging System**: Comprehensive training and testing logs with timestamps
- **Metrics Visualization**: Automatic generation of performance charts and comparison plots
- **Progress Tracking**: Real-time progress bars with detailed metrics using tqdm
- **Multi-dataset Evaluation**: Support for testing on 5 different polyp datasets
- **Automated Workflow**: Complete automation from training to testing and result tracking
- **Model Management**: Auto-naming conventions and model organization system
- **Extended Metrics**: Comprehensive evaluation including confidence intervals

### Loss Functions

- **Structure Loss** (default): Combination of Focal Loss + weighted IoU Loss
- **Dice Loss**: Standard Dice coefficient loss for medical segmentation
- **Tversky Loss**: Configurable α and β parameters for handling class imbalance
- **Combo Loss**: Combination of Dice and BCE losses
- **Boundary Loss**: Gradient-based boundary preservation
- **Unified Focal Loss**: Advanced focal loss with delta parameter

## Project Structure

```
Colon_Segment/
├── mmseg/                   # Core segmentation modules
│   ├── apis/                # API functions for training/testing
│   ├── core/                # Core functionality
│   ├── datasets/            # Dataset implementations
│   ├── models/              # Model definitions
│   │   ├── backbones/       # Backbone networks
│   │   ├── decode_heads/    # Decoding head components
│   │   ├── losses/          # Loss functions
│   │   ├── segmentors/      # Segmentor implementations
│   │   │   └── lib/         # Custom libraries for ColonFormer
│   ├── ops/                 # Operators and functions
│   └── utils/               # Utility functions
├── Data/                    # Data directory (ignored by git)
│   ├── TrainDataset/        # Training data
│   │   ├── image/           # Training images
│   │   └── mask/            # Training masks
│   └── TestDataset/         # Test datasets
│       ├── Kvasir/          # Kvasir dataset
│       ├── ETIS-LaribPolypDB/ # ETIS dataset
│       ├── CVC-ColonDB/     # CVC-ColonDB dataset
│       ├── CVC-ClinicDB/    # CVC-ClinicDB dataset
│       └── CVC-300/         # CVC-300 dataset
├── snapshots/               # Model snapshots (ignored by git)
│   ├── [model_name_1]/      # Directory for each trained model
│   │   ├── last.pth         # Latest model checkpoint
│   │   ├── epoch_20.pth     # Epoch-specific checkpoints
│   │   └── final.pth        # Final model weights
├── logs/                    # Log files (ignored by git)
│   ├── train_*.log          # Training logs
│   └── test_*.log           # Testing logs
├── results/                 # Visual results (ignored by git)
│   └── compa.png            # Comparison visualization
├── test_results/            # Test result metrics
│   ├── test_results_summary.csv     # Summary results
│   └── test_results_detailed.json   # Detailed results
├── Explain/                 # Explanation materials
│   └── ColonFormer_Technical_Guide_VN.md  # Technical guide
├── train.py                 # Training script
├── test.py                  # Testing script
├── complete_auto_system.py  # Automated workflow system
├── auto_rename_snapshots.py # Model management utility
├── utils.py                 # Utility functions
├── gpu_check.py             # GPU verification utility
├── test_auto_detect.py      # Auto-detection testing
├── test_with_component_loss.py # Component loss testing
├── ColonFormer_Inference_Notebook.ipynb # Inference notebook
├── requirements.txt         # Dependency requirements
└── README.md                # This file
```

## Environment Setup

### System Requirements

```bash
conda create -n Colon python=3.9
conda activate Colon
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Libraries

- Python 3.9
- PyTorch 2.1.2+cu118
- MMCV 2.2.0
- timm 0.9.12
- albumentations (data augmentation)
- OpenCV 4.5.0+
- matplotlib, seaborn (visualization)
- tqdm (progress bars)
- numpy 1.26.4
- scipy (confidence intervals)
- pandas (results tracking)

## Dataset Preparation

### Download Datasets

1. **Training Dataset**:

   - Download from [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view)
   - Extract to `./Data/TrainDataset/`

2. **Testing Dataset**:
   - Download from [Google Drive](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)
   - Extract to `./Data/TestDataset/`

### Data Structure

Organize your data as follows:

```
Data/
├── TrainDataset/
│   ├── image/          # Training images
│   └── mask/           # Corresponding masks (binary polyp segmentation masks)
└── TestDataset/
    ├── Kvasir/         # Dataset 1
    │   ├── images/     # Test images
    │   └── masks/      # Ground truth masks
    ├── ETIS-LaribPolypDB/  # Dataset 2
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/    # Dataset 3
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/   # Dataset 4
    │   ├── images/
    │   └── masks/
    └── CVC-300/        # Dataset 5
        ├── images/
        └── masks/
```

## Usage

### Training

**Basic Training:**

```bash
python train.py --backbone b3 --train_path ./Data/TrainDataset
```

**Advanced Training with Custom Parameters:**

```bash
python train.py \
    --backbone b3 \
    --num_epochs 50 \
    --batchsize 16 \
    --init_lr 1e-4 \
    --train_path ./Data/TrainDataset \
    --resume_path ./snapshots/ColonFormer.../last.pth \
    --loss_type tversky \
    --tversky_alpha 0.7 \
    --tversky_beta 0.3
```

**Training Parameters:**

- `--backbone`: b1, b2, b3, b4, b5 (default: b3)
  - b1: ColonFormer-XS
  - b2: ColonFormer-S
  - b3: ColonFormer-L
  - b4: ColonFormer-XL
  - b5: ColonFormer-XXL
- `--num_epochs`: Number of epochs (default: 20)
- `--batchsize`: Batch size (default: 8)
- `--init_lr`: Learning rate (default: 1e-4)
- `--init_trainsize`: Training image size (default: 352)
- `--clip`: Gradient clipping value (default: 0.5)
- `--loss_type`: Loss function (structure, dice, tversky, combo, boundary, unified_focal)
- `--train_path`: Path to training dataset
- `--train_save`: Save folder name (auto-generated if not provided)
- `--resume_path`: Path to checkpoint for resuming training

### Testing

**Test on Single Dataset:**

```bash
python test.py \
    --backbone b3 \
    --weight ./snapshots/ColonFormerB3/last.pth \
    --test_dataset Kvasir
```

**Test on All Datasets:**

```bash
python test.py \
    --backbone b3 \
    --weight ./snapshots/ColonFormerB3/last.pth \
    --test_dataset all
```

**Auto-test Untested Models:**

```bash
python test.py --auto_test --test_path ./Data/TestDataset
```

**Show All Results:**

```bash
python test.py --show_all
```

**Compare Models:**

```bash
python test.py --compare
```

**Testing Parameters:**

- `--backbone`: Backbone version used in training
- `--weight`: Path to trained checkpoint
- `--test_path`: Path to test datasets (default: ./Data/TestDataset)
- `--test_dataset`: Specific dataset (Kvasir, ETIS-LaribPolypDB, CVC-ColonDB, CVC-ClinicDB, CVC-300) or 'all'

### Automated Workflow System

**Complete Auto System:**

```bash
python complete_auto_system.py --show_status
python complete_auto_system.py --complete_workflow
```

**Auto-rename Snapshots:**

```bash
python auto_rename_snapshots.py --skip_empty --execute
```

The system automatically:

- Generates meaningful model names based on training parameters
- Organizes snapshots with consistent naming conventions
- Tracks tested vs untested models
- Provides comprehensive result summaries

### Output Files and Directories

**Training Outputs:**

- `logs/train_[model_name]_[timestamp].log`: Detailed training logs
- `snapshots/[model_name]/last.pth`: Latest checkpoint with metadata
- Training progress visualization and metrics tracking

**Testing Outputs:**

- `logs/test_[dataset]_[timestamp].log`: Detailed testing logs
- `test_results/test_results_summary.csv`: Comprehensive results database
- `results/test_results_[timestamp].png`: Performance visualization charts
- Console output with real-time metrics and progress bars

## Model Architecture

ColonFormer implements a sophisticated architecture with the following components:

### 1. Mix Transformer Backbone

- Hierarchical vision transformer for multi-scale feature extraction
- Support for 5 different model sizes (B1-B5)
- Efficient patch embedding and positional encoding

### 2. Context Feature Pyramid (CFP) Module

```python
self.CFP_1 = CFPModule(128, d=8)  # For x2 features
self.CFP_2 = CFPModule(320, d=8)  # For x3 features
self.CFP_3 = CFPModule(512, d=8)  # For x4 features
```

### 3. Axial Attention Mechanism

```python
self.aa_kernel_1 = AA_kernel(128, 128)  # 44x44 resolution
self.aa_kernel_2 = AA_kernel(320, 320)  # 22x22 resolution
self.aa_kernel_3 = AA_kernel(512, 512)  # 11x11 resolution
```

### 4. Three-Stage Reverse Attention

- **Stage 1**: 512→320 features with 32x upsampling
- **Stage 2**: 320→128 features with 16x upsampling
- **Stage 3**: 128→output with 8x upsampling
- Each stage uses reverse attention: `(-1 * sigmoid(decoder) + 1) * features`

### 5. Multi-output Architecture

Returns 4 outputs at different scales:

- `lateral_map_5`: Main output (8x upsampling)
- `lateral_map_3`: Intermediate output (16x upsampling)
- `lateral_map_2`: Intermediate output (32x upsampling)
- `lateral_map_1`: Coarse output (4x upsampling)

## Loss Functions

### Structure Loss (Default)

Combines Focal Loss and weighted IoU Loss for handling class imbalance and boundary accuracy.

### Advanced Loss Options

```bash
# Tversky Loss - better for small objects
--loss_type tversky --tversky_alpha 0.7 --tversky_beta 0.3

# Dice Loss - standard medical segmentation
--loss_type dice

# Combo Loss - BCE + Dice combination
--loss_type combo --combo_alpha 0.5

# Boundary Loss - gradient-based boundary preservation
--loss_type boundary --boundary_theta0 3 --boundary_theta 5

# Unified Focal Loss - advanced focal loss
--loss_type unified_focal --focal_gamma 2.0 --focal_delta 0.6
```

## Performance Metrics

### Comprehensive Evaluation

- **Dice Coefficient**: Overlap measure for segmentation quality
- **Mean IoU (mIoU)**: Intersection over Union average
- **Precision**: Positive predictive value
- **Recall/Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Accuracy**: Overall correctness
- **F1-Score**: Harmonic mean of precision and recall

### Statistical Analysis

- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Per-dataset Analysis**: Detailed breakdown by test dataset
- **Model Comparison**: Comprehensive performance comparison charts
- **Progress Tracking**: Real-time metric monitoring during training/testing

## Automated Features

### Model Management

- **Auto-naming**: Generates descriptive names based on training parameters
- **Metadata Tracking**: Stores training configuration in checkpoints
- **Organization**: Automatic folder structure and file organization

### Testing Automation

- **Untested Model Detection**: Automatically finds models that haven't been tested
- **Batch Testing**: Tests multiple models sequentially
- **Result Aggregation**: Consolidates results across all models and datasets

### Visualization

- **Performance Charts**: Automatic generation of comparison plots
- **Radar Charts**: Multi-metric visualization
- **Progress Plots**: Training progress and convergence analysis

## Model Variants and Performance

| Model           | Parameters | Backbone | Recommended Use                   |
| --------------- | ---------- | -------- | --------------------------------- |
| ColonFormer-XS  | ~13M       | MIT-B1   | Fast inference, limited resources |
| ColonFormer-S   | ~25M       | MIT-B2   | Balanced speed/accuracy           |
| ColonFormer-L   | ~47M       | MIT-B3   | Standard configuration            |
| ColonFormer-XL  | ~62M       | MIT-B4   | High accuracy requirements        |
| ColonFormer-XXL | ~82M       | MIT-B5   | Maximum performance               |

## Changes from Original Version

### Compatibility Updates

- Updated for MMCV 2.2.0 compatibility
- Fixed import statements and initialization functions
- Added fallback implementations for deprecated modules
- Replaced deprecated `F.upsample` with `F.interpolate`

### Enhanced Features

- **Complete Automation**: End-to-end workflow automation
- **Advanced Metrics**: Extended evaluation with statistical analysis
- **Model Management**: Sophisticated naming and organization system
- **Multiple Loss Functions**: 6 different loss function options
- **Real-time Monitoring**: Progress bars and live metric updates

### Performance Improvements

- Optimized training loop with better memory management
- Enhanced data loading with proper error checking
- Improved checkpoint saving with metadata
- Parallel testing capabilities for multiple datasets

## Technical Requirements

### Hardware

- NVIDIA GPU with CUDA support (recommended: RTX 3080 or better)
- Minimum 8GB GPU memory for batch size 8
- 16GB+ system RAM recommended
- SSD storage for faster data loading

### Software

- CUDA 11.8 or compatible
- cuDNN for accelerated training
- Python 3.9+ environment
- Windows 10/11 or Linux support

## Advanced Usage

### Jupyter Notebook Integration

- `ColonFormer_Inference_Notebook.ipynb`: Interactive inference and visualization
- Use this notebook for quick testing and visualization of model outputs

### GPU Verification

```bash
# Check if GPU is available and properly configured
python gpu_check.py
```

### Result Analysis

```bash
# View detailed result analysis
python test.py --show_all

# Compare specific models
python test.py --compare

# Export results to CSV
# Results automatically saved to test_results/test_results_summary.csv
```

### Custom Training Configurations

```bash
# Training with custom loss combination
python train.py \
    --backbone b4 \
    --loss_type tversky \
    --tversky_alpha 0.8 \
    --tversky_beta 0.2 \
    --num_epochs 100 \
    --batchsize 32 \
    --init_lr 5e-5
```

### Common Issues and Fixes

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Try a smaller backbone (b1 or b2)
   - Reduce image size with `--init_trainsize`

2. **Dataset Loading Errors**:
   - Verify dataset paths and folder structure
   - Check file permissions
   - Ensure image and mask files match correctly

3. **Training Instability**:
   - Try gradient clipping (`--clip 0.5`)
   - Reduce learning rate
   - Try different loss functions

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{duc2022colonformer,
  title={Colonformer: An efficient transformer based method for colon polyp segmentation},
  author={Duc, Nguyen Thanh and Oanh, Nguyen Thi and Thuy, Nguyen Thi and Triet, Tran Minh and Dinh, Viet Sang},
  journal={IEEE Access},
  volume={10},
  pages={80575--80586},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgments

- Thanks to the original authors [@ducnt9907](https://github.com/ducnt9907) for the ColonFormer implementation
- Original repository: [ducnt9907/ColonFormer](https://github.com/ducnt9907/ColonFormer)
- Paper: "ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation" - IEEE Access 2022

## License

This project is distributed under the license from the original repository. See [LICENSE](LICENSE) for more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Issues and Support

If you encounter any problems or have questions:

1. Check the existing issues in the repository
2. Create a new issue with detailed information about your problem
3. Include system information, error messages, and steps to reproduce
4. Check the logs in the `logs/` directory for detailed error information
