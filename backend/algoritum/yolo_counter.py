from ultralytics import YOLO
import cv2

def count_chromatin_with_yolo(model, image_path):
    """
    ใช้ YOLOv8 นับจำนวน Object (Chromatin) ในภาพ
    Return: (จำนวนจุดที่พบ, รายการพิกัด [ [x1, y1, x2, y2], ... ])
    """
    try:
        # Run inference
        # conf=0.25 คือค่าความมั่นใจขั้นต่ำ (ปรับได้ตามความเหมาะสม)
        results = model.predict(source=image_path, conf=0.25, verbose=False)
        
        # ดึงข้อมูล Boxes ที่โมเดลเจอ
        boxes = results[0].boxes
        count = len(boxes)
        
        # ✨ สร้าง List เก็บพิกัดของแต่ละกล่อง
        boxes_list = []
        for box in boxes:
            # ดึงพิกัด (x1, y1, x2, y2) และแปลงเป็น List ธรรมดา (เพื่อให้ส่งเป็น JSON ได้)
            coords = box.xyxy[0].tolist() 
            boxes_list.append(coords)
            
        # ส่งคืนค่า 2 ตัว: (จำนวน, รายการพิกัด)
        return count, boxes_list

    except Exception as e:
        print(f"⚠️ YOLO Counting Error: {e}")
        # กรณี Error ให้ส่งคืน 0 และ List ว่าง
        return 0, []