import cv2
import numpy as np
import os
import sys
import shutil

# เพิ่ม Path เพื่อหาไฟล์ cellree.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import cellree 
except ImportError:
    print("🚨 Error: ไม่พบไฟล์ cellree.py ในโฟลเดอร์")

def get_diameter_and_visualize(image_path, save_viz_path=None):
    """
    วัดขนาด Diameter โดยใช้กระบวนการที่ทนทานต่อภาพฟิล์มเลือด
    แก้ปัญหา 0 px โดยการปรับ Threshold ให้ครอบคลุมทั้งเซลล์
    """
    img = cv2.imread(image_path)
    if img is None: return 0

    h, w = img.shape[:2]
    
    # 1. เตรียมภาพ: Gray -> GaussianBlur เพื่อลด Noise ภายในตัวเซลล์
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) 
    
    # 2. Adaptive Threshold: ใช้ Block Size 51 (กว้างขึ้น) 
    # เพื่อให้ข้ามรายละเอียดเชื้อภายในและจับขอบนอกของ RBC ได้
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 2)

    # 3. Morphology Close: เชื่อมช่องว่างที่ขาดให้ติดกันเป็นก้อนเดียว
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. หาเส้นขอบ (Contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if save_viz_path: cv2.imwrite(save_viz_path, img)
        return 0

    # 5. เลือก Contour ที่ใหญ่ที่สุด และกรองขยะ (ต้องมีขนาด > 15% ของรูป Crop)
    main_cell_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_cell_contour)
    
    if area < (h * w * 0.15):
        if save_viz_path: cv2.imwrite(save_viz_path, img)
        return 0

    # 6. คำนวณ Diameter จากพื้นที่จริง (Area-based) เพื่อความเสถียร
    # สูตร: Diameter = 2 * sqrt(Area / pi)
    diameter = 2 * np.sqrt(area / np.pi)
    
    # 7. วาดภาพ Visualization (เหลือแค่เส้นขอบเขียว)
    if save_viz_path:
        viz_img = img.copy()
        
        # วาดเส้นขอบสีเขียว (Contour จริง)
        cv2.drawContours(viz_img, [main_cell_contour], -1, (0, 255, 0), 2)
        
        # --- ส่วนที่ปิดการทำงาน (เอาวงกลมเหลืองออก) ---
        # ((x, y), radius) = cv2.minEnclosingCircle(main_cell_contour)
        # cv2.circle(viz_img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        
        cv2.imwrite(save_viz_path, viz_img)

    return diameter

def calculate_refined_baseline(baseline_diameters):
    """คำนวณค่าเฉลี่ย RBC ปกติโดยใช้ Median เพื่อตัดค่าที่ผิดปกติออกอัตโนมัติ"""
    if not baseline_diameters: return 50.0 
    return np.median(baseline_diameters)

def process_folder_sizes(case_folder_path):
    """
    วิเคราะห์ขนาดและรูปร่างเซลล์ในโฟลเดอร์ต่างๆ และส่งผลสรุปกลับไป
    """
    TARGET_FOLDERS = ["1chromatin", "band form", "basket form", "schuffner dot", "Appliqué"]
    possible_baseline = ["nomal_cell", "normal_cell"]
    baseline_path = None
    
    # หาโฟลเดอร์เซลล์ปกติเพื่อใช้เป็นค่าอ้างอิง (Baseline A)
    for name in possible_baseline:
        p = os.path.join(case_folder_path, name)
        if os.path.exists(p):
            baseline_path = p
            break
            
    VIZ_ROOT = os.path.join(case_folder_path, "size_visualization")
    os.makedirs(VIZ_ROOT, exist_ok=True)
    
    # --- Step 1: คำนวณ Baseline (A) ---
    baseline_diameters = []
    if baseline_path:
        for file in os.listdir(baseline_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_p = os.path.join(baseline_path, file)
                d = get_diameter_and_visualize(full_p)
                # เช็คความกลมด้วย cellree (Baseline ต้องกลม > 0.70)
                circ, _ = cellree.analyze_shape(full_p)
                if d > 0 and circ > 0.70:
                    baseline_diameters.append(d)
    
    baseline_A = calculate_refined_baseline(baseline_diameters)

    # --- Step 2: วิเคราะห์เชื้อ (B) ---
    results_summary = {} 
    amoeboid_count = 0 

    for folder_name in TARGET_FOLDERS:
        target_path = os.path.join(case_folder_path, folder_name)
        if not os.path.exists(target_path): continue
        
        # สร้างโฟลเดอร์เก็บภาพผลลัพธ์
        viz_folder = os.path.join(VIZ_ROOT, folder_name)
        os.makedirs(viz_folder, exist_ok=True)
            
        for file in os.listdir(target_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(target_path, file)
                viz_out = os.path.join(viz_folder, file)
                
                # 1. วัดขนาด B (และวาดรูป Viz)
                size_B = get_diameter_and_visualize(full_path, viz_out)
                
                # 2. วิเคราะห์รูปร่าง (เรียกใช้ cellree)
                circ, shape_stat = cellree.analyze_shape(full_path)
                
                if shape_stat == "Amoeboid":
                    amoeboid_count += 1
                    # เขียน Label บนรูป Visualization เพื่อตรวจสอบความแม่นยำ
                    tmp = cv2.imread(viz_out)
                    if tmp is not None:
                        cv2.putText(tmp, f"Amoeboid ({circ:.2f})", (5, 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        cv2.imwrite(viz_out, tmp)

                # คำนวณ Ratio (B/A)
                ratio = size_B / baseline_A if baseline_A > 0 else 0
                
                results_summary[file] = {
                    "folder": folder_name,
                    "size_px": round(size_B, 2),
                    "ratio": round(ratio, 2),
                    "size_status": "Enlarged" if ratio > 1.25 else "Normal",
                    "shape_status": shape_stat,
                    "circularity": round(circ, 4),
                    "viz_image": viz_out 
                }

    return results_summary, amoeboid_count