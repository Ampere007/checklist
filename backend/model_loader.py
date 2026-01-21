import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np

# --- 1. ฟังก์ชันโหลดโมเดล (คงเดิม) ---
def load_resnet_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Loading model to {device}...")

    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Model weights loaded from: {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return None, device

        model = model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, device

# --- 2. ✨ ฟังก์ชันใหม่: เช็คว่าเซลล์ "เรียบเนียน" เกินไปไหม ---
def is_cell_too_smooth(image_path):
    """
    ใช้ OpenCV เช็ค Texture ของภาพ
    ถ้าภาพเรียบเกินไป (Standard Deviation ต่ำ) แสดงว่าไม่มีเชื้อโรค (Parasite ต้องมีจุดสีเข้ม)
    """
    try:
        # อ่านภาพแบบขาวดำ
        img = cv2.imread(image_path, 0) 
        if img is None: return False
        
        # คำนวณค่าเบี่ยงเบนมาตรฐาน (Standard Deviation) ของสีในภาพ
        # ค่าต่ำ = ภาพเรียบๆ (เซลล์ปกติ)
        # ค่าสูง = ภาพมีจุดตัดกันชัดเจน (น่าจะมีเชื้อ)
        mean, std_dev = cv2.meanStdDev(img)
        score = std_dev[0][0]
        
        print(f"🔍 Texture Score for {os.path.basename(image_path)}: {score:.2f}")
        
        # ⚠️ เกณฑ์ตัดสิน: ถ้า Score ต่ำกว่า 20 แสดงว่าภาพเรียบมาก ไม่น่าใช่เชื้อ
        # (คุณอาจต้องปรับค่า 20 ขึ้นลงนิดหน่อยตามแสงของกล้องจุลทรรศน์)
        return score < 20.0 
        
    except Exception as e:
        print(f"Warning in texture check: {e}")
        return False

# --- 3. ฟังก์ชันทำนายผล (แก้ไขเพิ่ม Logic) ---
def predict_image_file(model, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ชื่อ Class ตามที่คุณกำหนด
    class_names = ['1chromatin', 'band form', 'basket form', 'nomal_cell', 'schuffner dot']
    NORMAL_CLASS = 'nomal_cell'

    try:
        # A. ให้ AI ทำนายก่อน
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_p, top_class = probs.topk(1, dim=1)
            
            confidence = top_p.item() * 100
            predicted_class = class_names[top_class.item()]

        # B. 🛡️ ด่านป้องกันที่ 1: Confidence Threshold
        # ถ้า AI ไม่มั่นใจ (ต่ำกว่า 85%) ปัดตกทันที
        if predicted_class != NORMAL_CLASS and confidence < 85.0:
            print(f"🛡️ AI Unsure ({confidence:.2f}%). Reverting {predicted_class} -> Normal.")
            return NORMAL_CLASS, confidence

        # C. 🛡️ ด่านป้องกันที่ 2: Texture Check (เฉพาะเคสที่เป็นเชื้อโรค)
        # ถ้า AI บอกว่าเป็นเชื้อ แต่ภาพดูเรียบเนียนผิดปกติ -> เชื่อ OpenCV ดีกว่า
        if predicted_class != NORMAL_CLASS:
            if is_cell_too_smooth(image_path):
                print(f"🛡️ Image too smooth. Reverting {predicted_class} -> Normal (Texture Check).")
                return NORMAL_CLASS, confidence

        return predicted_class, confidence

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        return "Unknown", 0.0