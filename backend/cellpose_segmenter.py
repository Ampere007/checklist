import cv2
import numpy as np
import os
import uuid
from cellpose import models, io
import traceback 

# --- 1. ส่วนโหลดโมเดล (ใช้แบบใหม่ที่โหลดทีหลัง) ---
cell_model = None

def get_cellpose_model():
    global cell_model
    if cell_model is None:
        try:
            print("⏳ Loading Cellpose model ('cyto2')... (This happens once)")
            cell_model = models.Cellpose(gpu=False, model_type='cyto2')
            print("✅ Cellpose model ('cyto2') loaded successfully.")
        except Exception as e:
            print(f"🚨 FATAL ERROR: Could not load Cellpose model: {e}")
            traceback.print_exc()
            cell_model = None
    return cell_model

def segment_and_save_cells(image_path):
    """
    ฟังก์ชันหลัก (แก้ไขสมบูรณ์):
    1. เรียก get_cellpose_model()
    2. (FIXED) อ่านไฟล์ภาพ image_path มาเก็บใน 'image_bgr'
    3. (FIXED) แก้ไข channels=[0, 0] และ diameter=None
    4. บันทึก "เฉพาะส่วนที่ Crop" (สี่เหลี่ยม, มีพื้นหลัง)
    """
    
    # --- 1. เรียกโหลดโมเดล ---
    try:
        model = get_cellpose_model() 
        if model is None:
            print("🚨 Cellpose model is not loaded or failed to load. Cannot segment.")
            return [] 
            
    except Exception as e:
        print(f"🚨 An unknown error occurred during model loading: {e}")
        traceback.print_exc()
        return []

    # --- 2. อ่านภาพและเตรียมโฟลเดอร์ ---
    try:
        # --- 🔴 [FIXED] เพิ่มโค้ดส่วนที่ขาดไปกลับเข้ามา ---
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"🚨 Error reading image: {image_path}")
            return []
        # ------------------------------------------------
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # --- 🔴 [FIX 2] แก้ไข Parameters สำหรับ Cellpose ---
        CHANNELS = [0, 0] # (ของใหม่) ใช้ Grayscale
        
        # 2.5 สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ (ตาม session)
        session_id = str(uuid.uuid4())
        output_dir = os.path.join('segmented_cells', session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Running Cellpose segmentation on {os.path.basename(image_path)}...")

        # --- 3. รัน Cellpose Model ---
        masks, flows, styles, diams = model.eval(
            image_rgb,
            diameter=None,        # (ของใหม่) ให้โมเดลคำนวณขนาดเอง
            channels=CHANNELS,    # (ของใหม่) ใช้ [0, 0]
            flow_threshold=0.1,
            cellprob_threshold=-1.0
        )

        saved_paths = []
        num_cells = masks.max() 
        
        if num_cells == 0:
            print("INFO: No cells found by Cellpose.")
            return []

        print(f"Cellpose found {num_cells} cells. Cropping and saving for GrabCut...")

        # --- 4. วนลูปเพื่อ "Crop" และ "บันทึก" ทีละเซลล์ ---
        for i in range(1, num_cells + 1):
            cell_mask = (masks == i) 
            
            y_indices, x_indices = np.where(cell_mask)
            if y_indices.size == 0:
                continue 
            
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            # 4.3. "ตัด" (Crop) จากภาพ BGR ต้นฉบับ (image_bgr)
            # *** บรรทัดนี้จะไม่ Error แล้ว เพราะเรามี image_bgr จาก cv2.imread แล้ว ***
            cropped_image = image_bgr[y_min:y_max+1, x_min:x_max+1]

            # --- 5. ✨ (FIXED) บันทึกภาพที่ Crop เลย (BGR 3 channels) ✨ ---
            output_filename = f"cell_crop_{i}.png" 
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
            saved_paths.append(output_path)

        print(f"✅ Saved {len(saved_paths)} cropped cells (with background).")
        # คืนค่า list ของไฟล์ที่ crop แล้ว
        return saved_paths 

    except Exception as e:
        print(f"🚨 An error occurred during segmentation: {e}")
        traceback.print_exc()
        return []