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
    ตัดภาพเซลล์ + ถมดำพื้นหลัง (Masking) เพื่อไม่ให้เซลล์ข้างๆ ติดมา
    """
    try:
        model = get_cellpose_model() 
        if model is None: return []

        image_bgr = cv2.imread(image_path)
        if image_bgr is None: return []
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = image_bgr.shape
        
        # ---------------------------------------------------------
        # 1. ปรับจูน Cellpose ให้แยกเซลล์ติดกันได้ดีขึ้น (แก้จากรอบที่แล้ว)
        # ---------------------------------------------------------
        masks, _, _, _ = model.eval(
            image_rgb, 
            diameter=45,          # 👈 ระบุขนาดเซลล์ (บังคับให้แยกก้อนใหญ่)
            channels=[0, 0],    
            flow_threshold=0.4,   # 👈 ค่ามาตรฐาน ช่วยเรื่อง Shape
            cellprob_threshold=0.0 # 👈 ช่วยให้มั่นใจว่าเป็นเซลล์
        )

        num_cells = masks.max()
        if num_cells == 0: return []

        saved_cells_data = []
        session_id = str(uuid.uuid4())
        output_dir = os.path.join('segmented_cells', session_id)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(1, num_cells + 1):
            # หาตำแหน่งของเซลล์หมายเลข i
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

            # BBox Data
            bbox = {
                "x": int(x_min),
                "y": int(y_min),
                "w": int(x_max - x_min),
                "h": int(y_max - y_min)
            }

            # ---------------------------------------------------------
            # ✨ 2. เทคนิคใหม่: Cut & Mask (ตัดแล้วถมดำรอบๆ)
            # ---------------------------------------------------------
            
            # 2.1 ตัดภาพมาแบบพอดีตัวก่อน (ยังไม่เผื่อ Padding เยอะ)
            # ดึงเฉพาะส่วน BBox ของเซลล์นั้นๆ
            cell_roi = image_bgr[y_min:y_max+1, x_min:x_max+1]
            mask_roi = masks[y_min:y_max+1, x_min:x_max+1]

            # 2.2 สร้าง Mask เฉพาะตัว (อะไรที่ไม่ใช่เลข i ให้เป็น 0)
            # ผลลัพธ์: พื้นหลังเป็นสีดำ, เซลล์เพื่อนบ้านเป็นสีดำ, ตัวเราเป็นสีขาว
            isolated_mask = np.zeros_like(mask_roi, dtype=np.uint8)
            isolated_mask[mask_roi == i] = 255

            # 2.3 เอา Mask ไปแปะลงบนรูปจริง (Bitwise AND)
            # ตอนนี้เซลล์เพื่อนบ้านในกรอบสี่เหลี่ยมจะหายไป กลายเป็นสีดำทันที!
            masked_cell = cv2.bitwise_and(cell_roi, cell_roi, mask=isolated_mask)

            # 2.4 (Optional) ใส่ Padding สีดำเพิ่ม เพื่อให้รูปไม่ดูอึดอัด
            # ใช้ copyMakeBorder ของ OpenCV จะเติมขอบด้วยสีดำอัตโนมัติ
            pad = 5 # เพิ่มขอบดำ 5px รอบๆ
            final_image = cv2.copyMakeBorder(
                masked_cell, pad, pad, pad, pad, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # Save
            output_filename = f"cell_crop_{i}.png" 
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, final_image)
            
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
    คัดกรองเซลล์ (ปรับค่า Limit ให้เหมาะสมกับการแยกเซลล์แฝด)
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
    
    # ปรับให้แคบลง เพื่อกันพวกเซลล์แฝด
    MIN_LIMIT = median_area * 0.4
    MAX_LIMIT = median_area * 2.0 
    
    for i, item in enumerate(cell_data_list):
        area = areas[i]
        path = item['file_path']

        if area < MIN_LIMIT or area > MAX_LIMIT:
            try: os.remove(path)
            except: pass
            continue
            
        valid_data.append(item)
        
    return valid_data