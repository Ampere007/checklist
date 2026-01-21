import os
import torch

print("----- เริ่มการตรวจสอบไฟล์โมเดล -----")
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📂 โฟลเดอร์ปัจจุบัน: {current_dir}")

target_filename = "/Users/ampere/Downloads/Aim/backend/model/best_resnet-50_v5_finetuned.pth"
found_path = None

# เดินหาไฟล์ในทุกซอกทุกมุมของโฟลเดอร์ backend
for root, dirs, files in os.walk(current_dir):
    if target_filename in files:
        found_path = os.path.join(root, target_filename)
        print(f"✅ เจอไฟล์แล้วที่: {found_path}")
        break

if found_path:
    print("⏳ กำลังทดสอบโหลดไฟล์...")
    try:
        # ลองโหลดเฉพาะ Weights ดูว่าไฟล์เสียไหม
        state_dict = torch.load(found_path, map_location='cpu')
        print("🎉 ไฟล์ใช้งานได้ปกติ! (Load Success)")
        print(f"👉 กรุณาก๊อปปี้ Path นี้ไปใส่ใน app.py บรรทัด VIT_MODEL_PATH:")
        print(f"'{found_path}'")
    except Exception as e:
        print(f"❌ ไฟล์มีปัญหา (Corrupted): {e}")
else:
    print(f"❌ หาไฟล์ '{target_filename}' ไม่เจอเลยในโฟลเดอร์นี้")
    print("👉 รบกวนเช็คว่าชื่อไฟล์สะกดถูกหรือไม่ หรือไฟล์ถูกลบไปแล้ว")