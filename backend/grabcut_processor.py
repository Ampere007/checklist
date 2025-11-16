# backend/grabcut_processor.py
import cv2
import numpy as np
import os
import glob
import traceback

def process_cells_with_grabcut(cropped_cell_paths):
    """
    ฟังก์ชันหลัก:
    1. รับ 'list' ของ path รูปเซลล์ที่ crop มา (จาก Cellpose)
    2. รัน GrabCut กับแต่ละรูป
    3. บันทึกผลลัพธ์ (พื้นหลังโปร่งใส) ในโฟลเดอร์ใหม่
    4. คืนค่า 'list' ของ path สุดท้าย
    """
    
    if not cropped_cell_paths:
        print("INFO [GrabCut]: No cell paths to process.")
        return []

    # --- 1. ตั้งค่าโฟลเดอร์ (สำคัญ!) ---
    # เราจะใช้ session_id เดิมจาก Cellpose
    # เช่น 'segmented_cells/abc-123/cell_crop_1.png' -> 'abc-123'
    try:
        # หา Path ของโฟลเดอร์ input (เช่น 'segmented_cells/abc-123')
        input_folder = os.path.dirname(cropped_cell_paths[0])
        # หา session_id (เช่น 'abc-123')
        session_id = os.path.basename(input_folder)
    except Exception as e:
        print(f"🚨 ERROR [GrabCut]: Could not determine session ID from path: {e}")
        return []

    # โฟลเดอร์สำหรับเก็บภาพที่ลบพื้นหลังแล้ว (แยกกัน)
    output_folder = os.path.join('grabcut_processed_cells', session_id) 

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"สร้างโฟลเดอร์ใหม่สำหรับ GrabCut ที่: {output_folder}")

    print(f"--- 🚀 Starting GrabCut Process ({len(cropped_cell_paths)} images) ---")
    
    final_saved_paths = [] # List ที่จะใช้ส่งคืน

    # --- 3. เริ่มต้นประมวลผลทีละภาพ (จาก List ที่รับมา) ---
    for file_path in cropped_cell_paths:
        filename = os.path.basename(file_path)
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"ข้ามไฟล์ {filename} เพราะอ่านไม่ได้")
                continue

            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            # ==== ขั้นตอนที่ 1: กำหนดกรอบ (โค้ดเดิมของคุณ) ====
            height, width = img.shape[:2]
            margin_x = int(width * 0.05) 
            margin_y = int(height * 0.05)
            # (เพิ่มเงื่อนไขกัน error ถ้าภาพเล็กมาก)
            rect_w = max(1, width - (margin_x * 2)) 
            rect_h = max(1, height - (margin_y * 2))
            rect = (margin_x, margin_y, rect_w, rect_h)

            # ==== ขั้นตอนที่ 2: รัน GrabCut (โค้ดเดิมของคุณ) ====
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

            # ==== ขั้นตอนที่ 3: สร้าง Mask สุดท้าย (โค้ดเดิมของคุณ) ====
            mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

            # ==== ขั้นตอนที่ 4: นำ Mask ไปใช้ (โค้ดเดิมของคุณ) ====
            result_bgr = cv2.bitwise_and(img, img, mask=mask2)
            alpha_channel = np.full(img.shape[:2], 255, dtype=np.uint8)
            alpha_channel[mask2 == 0] = 0 
            result_rgba = cv2.merge((result_bgr[:,:,0], result_bgr[:,:,1], result_bgr[:,:,2], alpha_channel))

            # ==== ขั้นตอนที่ 5: บันทึกไฟล์ (โค้ดเดิมของคุณ) ====
            output_filename = os.path.splitext(filename)[0] + "_processed.png"
            output_path = os.path.join(output_folder, output_filename)
            
            cv2.imwrite(output_path, result_rgba)
            #print(f"ประมวลผล GrabCut: {output_path}") # (อาจจะ log เยอะไป)
            
            # เพิ่ม path สุดท้ายนี้เข้าไปใน list
            final_saved_paths.append(output_path)

        except Exception as e:
            print(f"🚨 ERROR [GrabCut] ขณะรันไฟล์ {filename}: {e}")
            traceback.print_exc()
            continue

    print(f"--- ✅ GrabCut Process Finished. Saved {len(final_saved_paths)} cells. ---")
    
    # คืนค่า list ของไฟล์ที่ผ่าน GrabCut แล้ว
    return final_saved_paths