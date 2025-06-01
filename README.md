# ColonFormer: An Efficient Transformer-based Method for Colon Polyp Segmentation

An implementation of ColonFormer - an efficient colon polyp segmentation method using Transformer architecture with enhanced logging, visualization, and evaluation capabilities.

**Note**: This is a modified version compatible with MMCV 2.2.0. The original repository and official paper can be found at [ducnt9907/ColonFormer](https://github.com/ducnt9907/ColonFormer).

## Key Features

- **Mix Transformer (MiT) Backbone**: Support for MIT-B0 to MIT-B5 variants
- **Context Feature Pyramid (CFP) Module**: Multi-scale dilated convolution features
- **Axial Attention Mechanism**: Efficient axis-wise attention computation
- **Reverse Attention**: Attention mechanism for focus refinement
- **Multi-scale Output**: 4 feature maps at different resolutions
- **Advanced Logging System**: Comprehensive training and testing logs
- **Metrics Visualization**: Automatic generation of performance charts
- **Progress Tracking**: Real-time progress bars with detailed metrics
- **Multi-dataset Evaluation**: Support for testing on multiple datasets

## Environment Setup

### System Requirements

```bash
conda create -n Colon python=3.9
conda activate Colon
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install seaborn  # For visualization
```

### Core Libraries

- Python 3.9
- PyTorch 2.1.2+cu118
- MMCV 2.2.0
- timm 0.9.12
- albumentations
- OpenCV
- matplotlib, seaborn (for visualization)
- tqdm (for progress bars)

## Dataset Preparation

### Download Datasets

1. **Training Dataset**:

   - Download from [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view)
   - Extract to `./data/TrainDataset/`

2. **Testing Dataset**:
   - Download from [Google Drive](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)
   - Extract to `./data/TestDataset/`

### Data Structure

Organize your data as follows:

```
data/
├── TrainDataset/
│   ├── image/          # Training images
│   └── mask/           # Corresponding masks
└── TestDataset/
    ├── Kvasir/         # Dataset 1
    │   ├── images/
    │   └── masks/
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
python train.py --backbone b3 --train_path ./data/TrainDataset --train_save ColonFormerB3
```

**Advanced Training with Custom Parameters:**

```bash
python train.py \
    --backbone b3 \
    --num_epochs 50 \
    --batchsize 16 \
    --init_lr 1e-4 \
    --train_path ./data/TrainDataset \
    --train_save ColonFormerB3_v2 \
    --resume_path ./snapshots/ColonFormerB3/last.pth
```

**Training Parameters:**

- `--backbone`: b0, b1, b2, b3, b4, b5 (default: b3)
- `--num_epochs`: Number of epochs (default: 20)
- `--batchsize`: Batch size (default: 8)
- `--init_lr`: Learning rate (default: 1e-4)
- `--init_trainsize`: Training image size (default: 352)
- `--clip`: Gradient clipping value (default: 0.5)
- `--train_path`: Path to training dataset
- `--train_save`: Save folder name
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

**Testing Parameters:**

- `--backbone`: Backbone version used in training
- `--weight`: Path to trained checkpoint
- `--test_path`: Path to test datasets (default: ./data/TestDataset)
- `--test_dataset`: Specific dataset (Kvasir, ETIS-LaribPolypDB, CVC-ColonDB, CVC-ClinicDB, CVC-300) or 'all'

### Output Files

**Training Outputs:**

- `logs/train_[model_name]_[timestamp].log`: Detailed training logs
- `snapshots/[model_name]/last.pth`: Latest checkpoint
- `snapshots/[model_name]/`: Training progress snapshots

**Testing Outputs:**

- `logs/test_[dataset]_[timestamp].log`: Detailed testing logs
- `results/test_results_[timestamp].png`: Performance visualization
- Console output with real-time metrics

## Model Architecture

ColonFormer combines several key components:

1. **Mix Transformer Backbone**: Multi-scale feature extraction using hierarchical vision transformer
2. **Context Feature Pyramid (CFP) Module**: Context modeling with dilated convolutions at multiple scales
3. **Axial Attention**: Efficient long-range dependency capture along spatial axes
4. **Reverse Attention**: Boundary refinement through attention mechanism
5. **Multi-output Supervision**: Training with outputs at 4 different scales

### Loss Function

The model uses **Structure Loss** combining:

- **Focal Loss**: Handles class imbalance in medical segmentation
- **Weighted IoU Loss**: Emphasizes boundary accuracy with spatial weighting

## Performance Metrics

The evaluation includes comprehensive metrics:

- **Dice Coefficient**: Overlap measure for segmentation quality
- **Mean IoU (mIoU)**: Intersection over Union average
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate

All metrics are computed both overall and per-dataset for detailed analysis.

## Visualization Features

The system automatically generates:

- **Performance bar charts**: Overall metrics comparison
- **Dataset comparison plots**: Performance across different test sets
- **Radar charts**: Multi-metric performance visualization
- **Model information summary**: Architecture details and parameters

## Changes from Original Version

### Compatibility Updates

- Updated for MMCV 2.2.0 compatibility
- Fixed import statements and initialization functions
- Added fallback implementations for deprecated modules
- Replaced deprecated `F.upsample` with `F.interpolate`

### Enhanced Features

- **Comprehensive Logging**: Detailed logs for training and testing
- **Progress Bars**: Real-time training/testing progress with tqdm
- **Metrics Visualization**: Automatic chart generation and saving
- **Multi-dataset Support**: Enhanced testing on multiple datasets
- **Error Handling**: Robust error handling and informative messages
- **Training from Scratch**: Removed dependency on pretrained weights

### Performance Improvements

- Optimized training loop with better memory management
- Enhanced data loading with proper error checking
- Improved checkpoint saving and loading mechanisms

## Technical Requirements

### Hardware

- NVIDIA GPU with CUDA support (recommended: RTX 3080 or better)
- Minimum 8GB GPU memory for batch size 8
- 16GB+ system RAM recommended

### Software

- CUDA 11.8 or compatible
- cuDNN for accelerated training
- Python 3.9+ environment

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
