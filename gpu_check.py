# gpu_check.py
import os
import torch
import time
import numpy as np

def check_gpu_availability():
    print("="*50)
    print("KIỂM TRA KHẢ NĂNG SỬ DỤNG GPU")
    print("="*50)
    
    # Kiểm tra CUDA có khả dụng không
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Kiểm tra số lượng GPU
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs detected: {gpu_count}")
    
    # Kiểm tra thông tin từng GPU
    for i in range(gpu_count):
        print(f"\n--- GPU {i} Info ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")
    
    # Thử test GPU 0
    print("\n\n" + "="*50)
    print("KIỂM TRA HOẠT ĐỘNG GPU 0")
    print("="*50)
    
    try:
        # Chỉ định GPU 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
        
        # Tạo tensor trên GPU 0 và thực hiện phép tính đơn giản
        x = torch.rand(1000, 1000, device=device)
        start_time = time.time()
        for _ in range(10):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        gpu0_time = time.time() - start_time
        print(f"GPU 0 test completed in {gpu0_time:.4f} seconds")
        print("GPU 0 hoạt động bình thường")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Lỗi khi test GPU 0: {e}")
    
    # Thử test GPU 1
    print("\n\n" + "="*50)
    print("KIỂM TRA HOẠT ĐỘNG GPU 1")
    print("="*50)
    
    try:
        # Chỉ định GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda:0")  # Vẫn là cuda:0 vì sau khi set CUDA_VISIBLE_DEVICES=1, GPU 1 sẽ được đánh số lại là 0
        
        # Tạo tensor trên GPU 1 và thực hiện phép tính đơn giản
        x = torch.rand(1000, 1000, device=device)
        start_time = time.time()
        for _ in range(10):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        gpu1_time = time.time() - start_time
        print(f"GPU 1 test completed in {gpu1_time:.4f} seconds")
        print("GPU 1 hoạt động bình thường")
        
        # Nếu cả hai GPU đều chạy được, so sánh thời gian
        if 'gpu0_time' in locals():
            print(f"\nSo sánh tốc độ: GPU 0 / GPU 1 = {gpu0_time / gpu1_time:.2f}x")
    except Exception as e:
        print(f"Lỗi khi test GPU 1: {e}")
    
    # Test riêng cho model ColonFormer (nhẹ)
    print("\n\n" + "="*50)
    print("KIỂM TRA MODEL COLONFORMER TRÊN GPU 1")
    print("="*50)
    
    try:
        # Chỉ định GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        # Import ColonFormer model từ cấu trúc project
        from mmseg.models.segmentors import ColonFormer as UNet
        
        # Tạo model với backbone nhỏ nhất để test
        model = UNet(backbone=dict(
            type='mit_b1',  # Sử dụng backbone nhỏ nhất để test
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
        
        # Tạo input tensor
        dummy_input = torch.randn(1, 3, 352, 352).cuda()
        
        # Thực hiện forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        print(f"ColonFormer forward pass trên GPU 1 hoàn tất trong {forward_time:.4f} giây")
        print(f"Output shape: {output[0].shape}")
        print("GPU 1 có thể chạy model ColonFormer")
        
    except Exception as e:
        print(f"Lỗi khi test ColonFormer trên GPU 1: {e}")
    
    print("\n\n" + "="*50)
    print("KẾT LUẬN")
    print("="*50)
    
    if gpu_count >= 2 and 'gpu1_time' in locals():
        print("✅ Server có 2 GPU hoạt động bình thường")
        print("✅ Cả 2 GPU đều có thể sử dụng")
        print("✅ Bạn có thể sử dụng GPU 1 để chạy model của mình")
        print("\nĐể sử dụng GPU 1 trong code của bạn, thêm dòng sau vào đầu script:")
        print("import os")
        print("os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"")
        print("\nhoặc chạy lệnh với cú pháp:")
        print("CUDA_VISIBLE_DEVICES=1 python train.py [các tham số khác]")
    else:
        if gpu_count < 2:
            print("❌ Chỉ phát hiện được {gpu_count} GPU")
            print("❌ Không thể sử dụng GPU 1 vì nó không tồn tại")
        else:
            print("❌ Có vẻ như GPU 1 không hoạt động đúng cách")
            print("❌ Vui lòng liên hệ người quản trị hệ thống")

if __name__ == "__main__":
    check_gpu_availability()
