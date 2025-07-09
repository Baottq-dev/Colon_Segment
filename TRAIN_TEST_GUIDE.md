# ColonSegment – Hướng dẫn Train / Test chi tiết

> Áp dụng cho Windows 10/11 **hoặc** Linux, Python ≥ 3.9, GPU CUDA 11.x (đã kiểm thử với CUDA 11.8 + PyTorch 2.1).

---

## 1 • Chuẩn bị môi trường

```bash
# Bước 1 – Tạo virtual env (tuỳ chọn)
cd ColonSegment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / WSL
source venv/bin/activate

# Bước 2 – Cài thư viện
pip install -r requirements.txt
```

> Nếu GPU của bạn dùng CUDA khác 11.8, hãy thay liên kết whl trong `requirements.txt` (mục mmcv & torch) theo hướng dẫn của [MMCV](https://github.com/open-mmlab/mmcv#installation).  
> Thiếu Visual C++ (Windows) ➜ cài "Desktop development with C++".

---

## 2 • Chuẩn bị dữ liệu

```
YourDataset/
 ├─ images/   # ảnh RGB (.jpg|.png)
 └─ masks/    # mask nhị phân 0/255 – trùng tên ảnh
```

* Sửa tên thư mục thành `images` & `masks` (hoặc chỉnh hai dòng `glob` trong `train.py`).
* Kiểm tra nhanh:

```python
from glob import glob
root='YourDataset'
imgs=glob(f'{root}/images/*'); masks=glob(f'{root}/masks/*')
assert len(imgs)==len(masks)>0, 'Ảnh / mask không khớp!'
print('Số ảnh:',len(imgs))
```

---

## 3 • Tải weight pre-trained cho backbone (bắt buộc)

```bash
mkdir pretrained
curl -L -o pretrained/mit_b3.pth \
  https://github.com/NVlabs/SegFormer/releases/download/v0.0.0/mit_b3.pth
```

* B1 ➜ `mit_b1.pth`, B2 ➜ `mit_b2.pth`, …
* Mặc định `train.py` sẽ đọc: `pretrained/mit_<backbone>.pth`.

---

## 4 • Huấn luyện (Train)

### 4.1 Thiết lập seed (tuỳ chọn)
Thêm vào đầu `train.py` để tái lập kết quả:

```python
import random, numpy as np, torch
seed=42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

### 4.2 Lệnh train cơ bản (theo paper)

```bash
python train.py \
  --train_path YourDataset \
  --backbone b3 \
  --num_epochs 20 \
  --batchsize 8 \
  --init_lr 1e-4 \
  --loss_type structure
```

* Checkpoint & log sẽ nằm trong `snapshots/<tên_mô_hình>/` & `logs/`.

### 4.3 Tăng batch & LR

```bash
python train.py \
  --batchsize 16 \
  --init_lr 2e-4   # linear scaling rule
```

### 4.4 Loss nâng cao – Dice-Weighted-IoU-Focal

```bash
python train.py \
  --loss_type dice_weighted_iou_focal \
  --dice_weight 0.4 \
  --iou_weight 0.3 \
  --focal_weight 0.3 \
  --focal_alpha 0.25 \
  --focal_gamma 2.0
```

### 4.5 Resume training

```bash
python train.py --resume_path snapshots/<model>/last.pth
```

---

## 5 • Đánh giá (Test / Inference)

Chuẩn bị bộ test:

```
TestDataset/
 ├─ images/
 └─ masks/   # tuỳ chọn (để tính metrics)
```

### 5.1 Test đầy đủ

```bash
python test.py \
  --backbone b3 \
  --weights snapshots/<model>/final.pth \
  --test_path TestDataset \
  --trainsize 352
```

* Tính Dice, IoU, Acc, Sens, Spec, F1 … cho từng ảnh & tổng.  
* Lưu kết quả vào `results/` (hình) và `test_results/` (CSV, JSON).

### 5.2 Tự động test tất cả model chưa đánh giá

```bash
python test.py --auto_test True
```

### 5.3 Inference 1 ảnh nhanh

```python
import cv2, torch
from mmseg.models.segmentors import ColonFormer as UNet
model=UNet(..., pretrained=None).cuda()
ckpt=torch.load('snapshots/<model>/final.pth')
model.load_state_dict(ckpt['state_dict']); model.eval()
img=cv2.imread('sample.jpg'); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.resize(img,(352,352)).astype('float32')/255
img=torch.tensor(img.transpose(2,0,1))[None].cuda()
with torch.no_grad():
    mask=torch.sigmoid(model(img)[-1])[0,0].cpu().numpy()
cv2.imwrite('pred.png',(mask>0.5)*255)
```

---

## 6 • Theo dõi & Phân tích

* **Log**: `logs/<tên_model>.log` – chứa LR, Loss, Dice, IoU, thời gian.  
* **TensorBoard**: thêm `SummaryWriter` trong `train.py` nếu cần.  
* **Chẩn đoán OOM**: giảm `--batchsize` hoặc bật AMP (`torch.cuda.amp`).

---

## 7 • Khắc phục sự cố

| Lỗi | Nguyên nhân / Giải pháp |
|-----|-------------------------|
| `ModuleNotFoundError: mmcv._ext` | Cài sai CUDA so với wheel – tải đúng whl mmcv. |
| `CUDA out of memory` | Giảm batch-size, bật AMP hoặc dùng B2/B1. |
| Dice ~0 | Thiếu pretrained **hoặc** ảnh-mask không khớp tên. |
| `size mismatch` khi load weight | Dùng sai backbone (b2↔b3), khác decode_head. |

---

**Chúc bạn train vui vẻ!** 