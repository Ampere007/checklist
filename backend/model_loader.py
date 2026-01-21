import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_resnet_model(model_path, num_classes):
    """
    โหลดโมเดล ResNet-50 และคืนค่า model, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Loading model to {device}...")

    try:
        # 1. โหลดโครงสร้าง ResNet-50
        model = models.resnet50(weights=None)
        
        # 2. ปรับแก้ Output Layer ให้ตรงกับไฟล์โมเดล (.pth)
        # ⚠️ แก้ไขตรงนี้: เปลี่ยนจาก Linear ธรรมดา เป็น Sequential ที่มี Dropout
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # เพิ่ม Dropout (Index 0)
            nn.Linear(num_ftrs, num_classes) # ตัวนี้จะกลายเป็น Index 1 (ตรงกับ fc.1.weight)
        )

        # 3. โหลด Weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"✅ Model weights loaded successfully from: {model_path}")
        else:
            print(f"❌ Model file not found at: {model_path}")
            return None, device

        model = model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        # กรณีฉุกเฉิน: ถ้าโหลด Sequential ไม่ได้ ให้ลองแบบเดิม (เผื่อไฟล์อื่น)
        try:
            print("⚠️ Retrying with simple Linear layer...")
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            print("✅ Recovered with simple Linear layer!")
            return model, device
        except:
            return None, device

def predict_image_file(model, device, image_path):
    """
    รับ path รูปภาพ -> แปลงภาพ -> ส่งเข้าโมเดล -> คืนค่าชื่อ Class และ Confidence
    """
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Class Names
    class_names = ['1chromatin', 'band form', 'basket form', 'nomal_cell', 'schuffner dot']

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            top_p, top_class = probs.topk(1, dim=1)
            confidence = top_p.item() * 100
            predicted_class = class_names[top_class.item()]
            
            return predicted_class, confidence
    except Exception as e:
        print(f"⚠️ Error predicting image: {e}")
        return "Unknown", 0.0