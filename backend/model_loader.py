import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import glob
from ultralytics import YOLO
import cv2  # (สำคัญ) Import cv2 สำหรับวาดกรอบ
import numpy as np
import traceback

# ==========================================================
# ===== ฟังก์ชันสำหรับโมเดล ResNet50 (Schuffner, Basket) =====
# (อัปเดตตามโค้ดใหม่ของคุณ)
# ==========================================================
def get_transform():
    """สร้างชุดคำสั่งสำหรับแปลงรูปภาพก่อนเข้าโมเดล ResNet"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_classification_model(model_path, num_classes=2):
    """โหลดโมเดล ResNet-50 และใส่ weights ที่เทรนไว้"""
    try:
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        print(f"✅ ResNet Model '{os.path.basename(model_path)}' loaded successfully.")
        return model
    except Exception as e:
        print(f"🚨 ERROR loading ResNet model: {e}")
        traceback.print_exc()
        return None

def run_prediction(model, image_folder_path, transform, class_names):
    """นำภาพทั้งหมดในโฟลเดอร์มาทำนายผลด้วย ResNet และสรุปผล"""
    if not os.path.exists(image_folder_path):
        return {"total_cells": 0, "predictions": {name: 0 for name in class_names}, "found_paths": []}
    
    image_files = glob.glob(os.path.join(image_folder_path, '*.png'))
    if not image_files:
        return {"total_cells": 0, "predictions": {name: 0 for name in class_names}, "found_paths": []}
    
    predictions_summary = {name: 0 for name in class_names}
    found_paths = []
    found_class_name = 'found' # คลาสที่เราสนใจคือ 'found'

    for img_path in image_files:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted_idx = torch.max(outputs, 1)
            
            predicted_class_name = class_names[predicted_idx.item()]
            predictions_summary[predicted_class_name] += 1

            if predicted_class_name == found_class_name:
                found_paths.append(img_path)
        except Exception as e:
            print(f"Prediction error on file: {img_path}, Error: {e}")
            
    return {
        "total_cells": len(image_files), 
        "predictions": predictions_summary, 
        "found_paths": found_paths
    }


# =======================================================
# ===== ฟังก์ชันสำหรับโมเดล YOLOv8 (Chromatin) =====
# (อัปเดตตามตรรกะที่ถูกต้องจากครั้งก่อน)
# =======================================================
def load_yolo_model(model_path):
    """
    โหลดโมเดล YOLOv8 จากไฟล์ .pt
    """
    try:
        model = YOLO(model_path) # ใช้ class YOLO จาก ultralytics
        print(f"✅ YOLOv8 Model '{os.path.basename(model_path)}' loaded successfully.")
        return model
    except Exception as e:
        print(f"🚨 ERROR loading YOLO model: {e}")
        traceback.print_exc()
        return None

def run_yolo_prediction(model, edge_image_dir, color_image_dir, output_dir):
    """
    (FIXED) ตรรกะใหม่สำหรับ YOLO Object Detection
    1. รัน YOLO บน 'edge_image_dir' (ภาพขอบ)
    2. กรองเฉพาะภาพที่มี detection (แก้ปัญหาเจอ 4 เซลล์)
    3. โหลดภาพสีจาก 'color_image_dir'
    4. วาดกรอบลงบนภาพสี (แก้ปัญหาไม่มีกรอบ)
    5. บันทึกผลลัพธ์ลงใน 'output_dir'
    """
    summary = {"found": 0, "not_found": 0, "found_paths": []}
    if model is None: return summary
    
    # ดึงภาพ "ขอบ" ทั้งหมด
    edge_image_paths = glob.glob(os.path.join(edge_image_dir, 'edge_*.png'))
    
    for edge_path in edge_image_paths:
        try:
            # 1. รัน YOLO prediction บน "ภาพขอบ"
            results = model(edge_path, verbose=False, conf=0.25) # (ปรับ conf score ได้)
            result = results[0] # เอาผลลัพธ์แรก
            
            # 2. (สำคัญ) ตรวจสอบว่า "พบ" โครมาทินหรือไม่
            if len(result.boxes) > 0:
                # --- ถ้าพบ Detection ---
                summary["found"] += 1
                
                # 3. หา "ภาพสี" ที่ตรงกัน
                edge_filename = os.path.basename(edge_path)
                # "edge_cell_crop_1_processed.png" -> "cell_crop_1_processed.png"
                color_filename = edge_filename.replace('edge_', '')
                color_path = os.path.join(color_image_dir, color_filename)
                
                if not os.path.exists(color_path):
                    print(f"⚠️ Warning: ไม่พบภาพสีที่ {color_path}")
                    continue
                    
                # 4. โหลด "ภาพสี" เพื่อวาดกรอบ
                color_image = cv2.imread(color_path)
                
                # 5. วาดกรอบ (Bounding Boxes) ทั้งหมดที่พบบน "ภาพสี"
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    # วาดสี่เหลี่ยมสีเขียว (0, 255, 0) หนา 2 pixels
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 6. บันทึก "ภาพสีที่มีกรอบ"
                output_filename = f"yolo_{color_filename}"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, color_image)
                
                # 7. เพิ่ม Path ของ "ภาพใหม่ที่มีกรอบ" เข้าไปใน summary
                summary["found_paths"].append(output_path)
                
            else:
                # --- ถ้าไม่พบ Detection ---
                summary["not_found"] += 1
                
        except Exception as e:
            print(f"🚨 ERROR during YOLO prediction on {edge_path}: {e}")
            traceback.print_exc()
            
    # สร้าง 'predictions' summary ให้สอดคล้องกับ ResNet
    summary["predictions"] = {
        "found": summary["found"],
        "not_found": summary["not_found"]
    }
    
    # (เราคืนค่า summary ทั้งหมด แต่ app.py จะดึงไปใช้แค่บางส่วน)
    return summary