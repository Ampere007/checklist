import torch
import os

# ใส่ชื่อไฟล์โมเดลของคุณตรงนี้
model_path = "/Users/ampere/Downloads/Aim/backend/model/best_finetuned_vit_local.pth"

if os.path.exists(model_path):
    print(f"กำลังตรวจสอบไฟล์: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    print("-" * 30)
    print("🔍 รายชื่อ Layer หลักๆ ในไฟล์ .pth:")
    # ปริ้นเฉพาะส่วน Head เพื่อดูชื่อและขนาด
    for key, value in state_dict.items():
        if "head" in key or "fc" in key or "classifier" in key:
            print(f"ชื่อ Layer: {key} | ขนาด (Shape): {value.shape}")
            
    print("-" * 30)
else:
    print("❌ หาไฟล์ไม่เจอ เช็ค path ดีๆ นะ")