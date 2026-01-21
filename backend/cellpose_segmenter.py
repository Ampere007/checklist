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
    """
    ตัดภาพเซลล์และคืนค่าเป็น List ของ Dictionary
    Format: [{"id": 1, "file_path": "...", "bbox": {"x":..., "y":..., "w":..., "h":...}}]
    """
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

        saved_cells_data = [] # เปลี่ยนจากเก็บแค่ Path เป็นเก็บ Data ทั้งก้อน
        session_id = str(uuid.uuid4())
        output_dir = os.path.join('segmented_cells', session_id)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(1, num_cells + 1):
            cell_mask = (masks == i) 
            y_indices, x_indices = np.where(cell_mask)
            if y_indices.size == 0: continue 
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # ✨ Border Check: ยอมให้ติดขอบได้นิดนึง
            border_margin = 1 
            if (x_min <= border_margin or y_min <= border_margin or 
                x_max >= width - border_margin or y_max >= height - border_margin):
                continue 

            # --- [ส่วนสำคัญที่เพิ่มมา: เก็บพิกัดจริงเพื่อส่งหน้าเว็บ] ---
            # ต้องแปลงเป็น int ธรรมดา (ไม่ใช่ numpy int) เพื่อให้ JSON ไม่ error
            bbox = {
                "x": int(x_min),
                "y": int(y_min),
                "w": int(x_max - x_min),
                "h": int(y_max - y_min)
            }

            # --- การตัดภาพ (Padding) ---
            padding = 10 
            y_min_pad = max(0, y_min - padding)
            y_max_pad = min(height, y_max + padding)
            x_min_pad = max(0, x_min - padding)
            x_max_pad = min(width, x_max + padding)

            cropped_image = image_bgr[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            
            output_filename = f"cell_crop_{i}.png" 
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
            
            # เก็บข้อมูลลง List
            saved_cells_data.append({
                "id": i,
                "file_path": output_path,
                "bbox": bbox
            })

        return saved_cells_data 
    except Exception as e:
        print(f"Error in segmentation: {e}")
        return []

def filter_bad_cells(cell_data_list):
    """
    คัดกรองเซลล์ (รับ input เป็น List ของ Dictionary แล้ว)
    """
    if not cell_data_list: return []
    
    valid_data = []
    areas = []
    
    # คำนวณพื้นที่เพื่อหา Median (ใช้ bbox w*h หรือจะโหลดรูปมาก็ได้)
    # เพื่อความเร็ว ใช้ bbox คำนวณคร่าวๆ ได้เลย
    for item in cell_data_list:
        w = item['bbox']['w']
        h = item['bbox']['h']
        areas.append(w * h)
        
    if not areas: return []
    median_area = np.median(areas)
    
    # ✨ ตั้งค่ากว้างๆ ไว้ก่อน
    MIN_LIMIT = median_area * 0.3 
    MAX_LIMIT = median_area * 3.5 
    
    for i, item in enumerate(cell_data_list):
        area = areas[i]
        path = item['file_path']

        # เงื่อนไขการลบไฟล์
        if area < MIN_LIMIT: # ตัดขยะชิ้นเล็ก
            try: os.remove(path)
            except: pass
            continue
        if area > MAX_LIMIT: # ตัดก้อนใหญ่ยักษ์
            try: os.remove(path)
            except: pass
            continue
            
        # ถ้าผ่านเกณฑ์ ก็เก็บใส่ list
        valid_data.append(item)
        
    return valid_data