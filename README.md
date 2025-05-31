# ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation

Đây là implementation của ColonFormer - một phương pháp segmentation polyp đại tràng hiệu quả sử dụng Transformer architecture.

> **Lưu ý**: Đây là phiên bản đã được điều chỉnh để tương thích với MMCV 2.2.0. Repository gốc và paper chính thức có thể tìm thấy tại [ducnt9907/ColonFormer](https://github.com/ducnt9907/ColonFormer).

## 🎯 Tính năng chính

- **Mix Transformer (MiT) Backbone**: Sử dụng các phiên bản MIT-B0 đến MIT-B5
- **Context Feature Pyramid (CFP) Module**: Module đặc trưng với dilated convolution đa tỷ lệ
- **Axial Attention Mechanism**: Cơ chế attention theo trục cho hiệu quả tính toán
- **Reverse Attention**: Cơ chế attention ngược để tập trung vào vùng quan trọng
- **Multi-scale Output**: 4 feature maps ở các độ phân giải khác nhau

## 🔧 Môi trường

### Yêu cầu hệ thống

```bash
conda create -n Colon python=3.9
conda activate Colon
```

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Thư viện chính

- Python 3.9
- PyTorch 2.1.2+cu118
- MMCV 2.2.0
- timm 0.9.12
- albumentations
- OpenCV

## 📥 Tải dữ liệu

### Dataset cho training và testing

1. **Training Dataset**:

   - Tải từ [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view)
   - Giải nén và đặt vào thư mục `./data/TrainDataset/`

2. **Testing Dataset**:
   - Tải từ [Google Drive](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)
   - Giải nén và đặt vào thư mục `./data/TestDataset/`

## 📁 Cấu trúc dữ liệu

Tổ chức dữ liệu như sau:

```
data/
├── TrainDataset/
│   ├── image/          # Ảnh training
│   └── mask/           # Mask tương ứng
└── TestDataset/
    ├── image/          # Ảnh test
    └── mask/           # Mask tương ứng
```

## 🚀 Sử dụng

### Training

```bash
python train.py --backbone b3 --train_path ./data/TrainDataset --train_save ColonFormerB3
```

**Các tham số có sẵn:**

- `--backbone`: b0, b1, b2, b3, b4, b5 (mặc định: b3)
- `--num_epochs`: Số epochs (mặc định: 20)
- `--batchsize`: Batch size (mặc định: 8)
- `--init_lr`: Learning rate (mặc định: 1e-4)
- `--train_path`: Đường dẫn dữ liệu training
- `--train_save`: Tên folder lưu checkpoint

### Testing

```bash
python test.py --backbone b3 --weight ./snapshots/ColonFormerB3/last.pth --test_path ./data/TestDataset
```

**Các tham số:**

- `--backbone`: Phiên bản backbone đã training
- `--weight`: Đường dẫn checkpoint
- `--test_path`: Đường dẫn dữ liệu test

## 📊 Kiến trúc mô hình

ColonFormer kết hợp:

1. **Mix Transformer Backbone** để trích xuất multi-scale features
2. **CFP Module** cho context modeling với dilated convolutions
3. **Axial Attention** để capture long-range dependencies hiệu quả
4. **Reverse Attention** để refine segmentation boundaries

## 🔄 Thay đổi từ phiên bản gốc

- Cập nhật tương thích với MMCV 2.2.0
- Sửa lỗi import và initialization functions
- Thêm fallback implementations cho các module bị deprecated
- Tối ưu training loop với progress bars và metrics tracking

## 📝 Citation

Nếu sử dụng code này trong nghiên cứu, vui lòng cite paper gốc:

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

## 🙏 Acknowledgments

- Cảm ơn tác giả gốc [@ducnt9907](https://github.com/ducnt9907) cho implementation ColonFormer
- Repository gốc: [ducnt9907/ColonFormer](https://github.com/ducnt9907/ColonFormer)
- Paper: "ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation" - IEEE Access 2022

## 📄 License

Dự án này được phân phối dưới license từ repository gốc. Xem [LICENSE](LICENSE) để biết thêm chi tiết.
