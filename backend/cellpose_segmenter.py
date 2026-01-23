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
    ตัดภาพเซลล์แบบ 'Cookie Cutter' (แม่พิมพ์ตัดคุ้กกี้):
    1. สร้างภาพพื้นหลังสีชมพูเปล่าๆ รอไว้ (Canvas)
    2. ใช้ Mask ของเซลล์เป็นแม่พิมพ์ (ขยายขอบเล็กน้อย)
    3. 'ปั๊ม' เฉพาะตัวเซลล์จากภาพจริง ลงไปบน Canvas
    
    ข้อดี: รับประกัน 100% ว่าเพื่อนข้างบ้าน/ขยะ/เกล็ดเลือด จะไม่มีทางติดมา
           เพราะเราเลือกก๊อปปี้มาเฉพาะพื้นที่ของเซลล์เท่านั้น
    """
    try:
        model = get_cellpose_model() 
        if model is None: return []

        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return []
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = image_bgr.shape
        
        # ใช้ Settings แบบ Auto Diameter เพื่อให้เจอเซลล์ครบทุกขนาด
        masks, _, _, _ = model.eval(
            image_rgb, 
            diameter=None,        
            channels=[0, 0],    
            flow_threshold=0.4,   
            cellprob_threshold=0.0 
        )

        num_cells = masks.max()
        print(f"🔎 Cellpose found: {num_cells} cells") 

        if num_cells == 0: return []

        saved_cells_data = []
        session_id = str(uuid.uuid4())
        output_dir = os.path.join('segmented_cells', session_id)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(1, num_cells + 1):
            cell_indices = (masks == i)
            y_indices, x_indices = np.where(cell_indices)
            if y_indices.size == 0: continue 
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # Border Check
            border_margin = 1 
            if (x_min <= border_margin or y_min <= border_margin or 
                x_max >= width - border_margin or y_max >= height - border_margin):
                continue 

            bbox = {
                "x": int(x_min),
                "y": int(y_min),
                "w": int(x_max - x_min),
                "h": int(y_max - y_min)
            }

            # ---------------------------------------------------------
            # ✨ เทคนิค Cookie Cutter Strategy
            # ---------------------------------------------------------
            
            padding = 10
            y_start = max(0, y_min - padding)
            y_end = min(height, y_max + padding)
            x_start = max(0, x_min - padding)
            x_end = min(width, x_max + padding)

            # 1. เตรียมภาพต้นฉบับ (Source) และ Mask ของพื้นที่นี้
            roi_image = image_bgr[y_start:y_end, x_start:x_end]
            roi_mask = masks[y_start:y_end, x_start:x_end]

            # 2. คำนวณสีพื้นหลัง (Background Color) เพื่อเตรียมทำกระดาษเปล่า
            bg_pixels_mask = (roi_mask == 0)
            if np.sum(bg_pixels_mask) > 0:
                bg_color = roi_image[bg_pixels_mask].mean(axis=0).astype(np.uint8)
            else:
                bg_color = np.array([230, 230, 240], dtype=np.uint8) # สีชมพูมาตรฐาน

            # 3. สร้าง Canvas เปล่าๆ (กระดาษสีชมพู) ขนาดเท่า ROI
            # เริ่มต้นด้วยการเทสีพื้นหลังให้เต็มแผ่น
            final_roi = np.full_like(roi_image, bg_color)

            # 4. เตรียมแม่พิมพ์ (Mask) เฉพาะตัวเรา
            my_cell_mask = (roi_mask == i).astype(np.uint8)
            
            # ขยายขอบแม่พิมพ์ (Dilation) เพื่อให้ครอบคลุมขอบเซลล์และเชื้อที่เกาะขอบ
            # ใช้ค่า 4 เพื่อความปลอดภัยสำหรับ P. vivax
            mask_expansion = 3
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(my_cell_mask, kernel, iterations=mask_expansion)

            # 5. "ปั๊ม" ภาพลงไป (The Stamp) ✨
            # สั่งว่า: ตรงไหนที่เป็นรูแม่พิมพ์ (dilated_mask == 1) ให้เอาภาพจริงมาใส่
            # ส่วนตรงไหนที่ไม่ใช่ (เช่น เพื่อนบ้าน) ให้คงสีชมพูของ Canvas ไว้ตามเดิม
            # หมายเหตุ: [..., None] ใช้เพื่อให้ Dimension ตรงกับภาพสี (3 channels)
            final_roi = np.where(dilated_mask[..., None] == 1, roi_image, final_roi)
            
            # Save
            output_filename = f"cell_crop_{i}.png" 
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, final_roi)
            
            saved_cells_data.append({
                "id": i,
                "file_path": output_path,
                "bbox": bbox
            })

        return saved_cells_data 
    except Exception as e:
        print(f"Error in segmentation: {e}")
        traceback.print_exc()
        return []

def filter_bad_cells(cell_data_list):
    """
    คัดกรองเซลล์ (ใช้ Logic เดิมที่ดีอยู่แล้ว)
    """
    if not cell_data_list: return []
    valid_data = []
    areas = []
    
    for item in cell_data_list:
        w = item['bbox']['w']
        h = item['bbox']['h']
        areas.append(w * h)
        
    if not areas: return []
    median_area = np.median(areas)
    
    # Range กว้างๆ ไว้ก่อน
    MIN_LIMIT = median_area * 0.2
    MAX_LIMIT = median_area * 3.5 
    
    for i, item in enumerate(cell_data_list):
        area = areas[i]
        path = item['file_path']
        if area < MIN_LIMIT or area > MAX_LIMIT:
            try: os.remove(path)
            except: pass
            continue
        valid_data.append(item)
    
    return valid_data