import cv2
import numpy as np
import os
import uuid
from cellpose import models
import traceback 

cell_model = None

def get_cellpose_model():
    global cell_model
    if cell_model is None:
        try:
            print("⏳ Loading Cellpose model ('cyto2')...")
            cell_model = models.Cellpose(gpu=False, model_type='cyto2')
        except Exception as e:
            print(f"🚨 FATAL ERROR: {e}")
            cell_model = None
    return cell_model

def segment_and_save_cells(image_path):
    # ฟังก์ชันนี้กวาดมาให้หมด (Segment All)
    try:
        model = get_cellpose_model() 
        if model is None: return []

        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return []
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = image_bgr.shape
        
        masks, _, _, _ = model.eval(
            image_rgb, diameter=None, channels=[0, 0],    
            flow_threshold=0.1, cellprob_threshold=-1.0
        )

        num_cells = masks.max()
        if num_cells == 0: return []

        saved_paths = []
        session_id = str(uuid.uuid4())
        output_dir = os.path.join('segmented_cells', session_id)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(1, num_cells + 1):
            cell_mask = (masks == i) 
            y_indices, x_indices = np.where(cell_mask)
            if y_indices.size == 0: continue 
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # ✨ Border Check: ยอมให้ติดขอบได้นิดนึง (เผื่อเชื้ออยู่ริม)
            border_margin = 1 
            if (x_min <= border_margin or y_min <= border_margin or 
                x_max >= width - border_margin or y_max >= height - border_margin):
                continue 

            padding = 10 
            y_min_pad = max(0, y_min - padding)
            y_max_pad = min(height, y_max + padding)
            x_min_pad = max(0, x_min - padding)
            x_max_pad = min(width, x_max + padding)

            cropped_image = image_bgr[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            
            output_filename = f"cell_crop_{i}.png" 
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
            saved_paths.append(output_path)

        return saved_paths 
    except: return []

def filter_bad_cells(cell_paths):
    # ฟังก์ชันคัดกรอง (Relaxed Mode)
    if not cell_paths: return []
    
    valid_paths = []
    areas = []
    
    for path in cell_paths:
        img = cv2.imread(path)
        if img is None: continue
        h, w, _ = img.shape
        areas.append(h * w)
        
    if not areas: return []
    median_area = np.median(areas)
    
    # ✨ ตั้งค่ากว้างๆ ไว้ก่อน เพื่อไม่ให้ Schuffner หลุด
    MIN_LIMIT = median_area * 0.3 
    MAX_LIMIT = median_area * 3.5 # ยอมรับเซลล์ใหญ่ได้ถึง 3.5 เท่า
    
    for i, path in enumerate(cell_paths):
        area = areas[i]
        if area < MIN_LIMIT: # ตัดขยะชิ้นเล็ก
            try: os.remove(path)
            except: pass
            continue
        if area > MAX_LIMIT: # ตัดก้อนใหญ่ยักษ์จริงๆ
            try: os.remove(path)
            except: pass
            continue
            
        # ❌ ไม่เช็คสีม่วงแล้ว (No WBC Check) เพื่อรักษา Schuffner
        valid_paths.append(path)
        
    return valid_paths