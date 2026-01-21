import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import traceback
import cv2  # ✨ ต้องมี OpenCV สำหรับเช็คสี
import numpy as np # ✨ ต้องมี Numpy คำนวณค่าสี
from image_processor import preprocess_image_with_mask 

# ==========================================================
# ⚙️ CONFIG
# ==========================================================
# ชื่อไฟล์โมเดล (ใช้ชื่อไฟล์อย่างเดียว เพื่อให้ Path Relative ทำงานได้ถูกต้อง)
MODEL_FILENAME = 'best_resnet-50_new_start.pth'

CLASS_NAMES = ['1chromatin', 'band form', 'basket form', 'nomal_cell', 'schuffner dot']

# ==========================================================
# 1. Helper Function: เช็คสีม่วง (HSV Color Filter) 🎨
# ==========================================================
def is_color_intense_enough(image_path):
    """
    ฟังก์ชันตรวจสอบว่าในภาพมี 'เม็ดสีม่วง' (Chromatin) ที่เข้มพอหรือไม่
    ใช้สำหรับกรอง Noise จางๆ ที่ AI ชอบทายผิดว่าเป็นเชื้อ
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return True # ถ้าอ่านไฟล์ไม่ได้ ให้ปล่อยผ่านไปหา AI ดีกว่า
        
        # 1. แปลงเป็นระบบสี HSV (Hue, Saturation, Value)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. กำหนดช่วงสีม่วง/ชมพูเข้ม (สีของ Chromatin/Giemsa Stain)
        # Lower: ม่วงโทนกลางๆ
        # Upper: ม่วงเข้ม/ชมพูเข้ม
        lower_purple = np.array([120, 20, 30])  
        upper_purple = np.array([170, 255, 180]) 

        # 3. สร้าง Mask หาพื้นที่ที่เป็นสีม่วง
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # 4. นับจำนวนพิกเซลที่เป็นสีม่วงจริงๆ
        purple_pixel_count = cv2.countNonZero(mask)
        
        # ✨ THRESHOLD: ถ้ามีจุดสีม่วงน้อยกว่า 10 pixels แสดงว่าเป็นแค่ Noise หรือรอยเปื้อน
        if purple_pixel_count < 10: 
            # print(f"🔍 Color Check: Found only {purple_pixel_count} purple pixels (Too faint).")
            return False # ไม่ผ่านเกณฑ์ (จางไป)
        
        return True # ผ่านเกณฑ์ (มีสีม่วงชัดเจน)

    except Exception as e:
        print(f"⚠️ Color check error: {e}")
        return True # ถ้า Error ให้ยอมให้ผ่านไปก่อน

# ==========================================================
# 2. Image Transforms
# ==========================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ==========================================================
# 3. ResNet Model Loader
# ==========================================================
def load_resnet_model(model_path=None, num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ใช้ Relative Path เพื่อความชัวร์ (ไม่ต้องแก้ Path เวลาเปลี่ยนเครื่อง)
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ระบบจะหาที่ backend/model/best_resnet-50_new_start.pth
        model_path = os.path.join(current_dir, 'model', os.path.basename(MODEL_FILENAME))

    print(f"⏳ กำลังโหลดโมเดลจาก: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: ไม่พบไฟล์โมเดลที่ {model_path}")
        return None, device

    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),                 
            nn.Linear(num_ftrs, num_classes) 
        )

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval() 
        
        print(f"✅ โหลดโมเดลสำเร็จ! (Device: {device})")
        return model, device

    except Exception as e:
        print(f"🚨 CRITICAL ERROR โหลดโมเดลไม่ผ่าน: {e}")
        traceback.print_exc()
        return None, device

# ==========================================================
# 4. Prediction Function (With Color Check Logic)
# ==========================================================
def predict_image_file(model, device, image_path):
    try:
        if model is None:
            return "Model Error", 0.0

        # ✨ STEP 0: กรองด้วยสี (Color Filter) ก่อนเลย ✨
        # ถ้าสีม่วงจางเกินไป บังคับตอบ Normal ทันที ไม่ต้องถาม AI
        if not is_color_intense_enough(image_path):
            print(f"🎨 Color Check Failed for {os.path.basename(image_path)} (Faint Stain). Force Normal.")
            return 'nomal_cell', 100.0

        # ✨ STEP 1: เรียกใช้ฟังก์ชันปรับภาพ (Masking)
        img = preprocess_image_with_mask(image_path)
        
        if img is None:
            return "Image Error", 0.0

        # STEP 2: แปลงรูปส่งเข้า AI
        transform = get_transform()
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        class_idx = predicted_idx.item()
        conf_score = confidence.item() * 100 
        result_class = CLASS_NAMES[class_idx]
        
        return result_class, conf_score

    except Exception as e:
        print(f"⚠️ Error predicting {image_path}: {e}")
        return "Error", 0.0

if __name__ == "__main__":
    model, dev = load_resnet_model()
    if model:
        print("🎉 ระบบ Model Loader พร้อมใช้งาน!")