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
    วัดขนาด Diameter โดยใช้ Convex Hull เพื่อแก้ปัญหาขอบเซลล์แหว่ง
    ทำให้ได้ขนาดที่แท้จริง (Equivalent Diameter)
    """
    img = cv2.imread(image_path)
    if img is None: return 0

    h, w = img.shape[:2]
    
    # 1. เตรียมภาพ: Gray -> Blur (เพิ่มขนาด Kernel เพื่อลด Noise ผิวเซลล์)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0) 
    
    # 2. Otsu's Thresholding: แยก Background ออกจาก Cell
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Morphology: ถมรูพรุนภายในและเชื่อมขอบ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. หาเส้นขอบ (Contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if save_viz_path: cv2.imwrite(save_viz_path, img)
        return 0

    # 5. เลือก Contour ที่ "อยู่ใกล้กลางภาพ" มากที่สุด 
    # (ป้องกันการไปจับขอบ Bounding Box หรือขยะที่มุมภาพ)
    center_img = (w // 2, h // 2)
    best_contour = None
    min_dist = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.05): continue # กรองขยะที่มีขนาดเล็กกว่า 5% ของรูป
        
        # หาจุดศูนย์กลางของ Contour (Centroid)
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # คำนวณระยะห่างจากกลางภาพ
        dist = np.sqrt((cX - center_img[0])**2 + (cY - center_img[1])**2)
        if dist < min_dist:
            min_dist = dist
            best_contour = cnt

    if best_contour is None:
        if save_viz_path: cv2.imwrite(save_viz_path, img)
        return 0

    # --- ✨ KEY FIX: ใช้ Convex Hull หาขอบเขตจริง ---
    # ช่วยแก้ปัญหาขอบหยัก หรือรอยแหว่งจากการ Threshold 
    hull = cv2.convexHull(best_contour)

    # 6. คำนวณ Diameter จากพื้นที่ของ Convex Hull (Area-based)
    # สูตร: Diameter = 2 * sqrt(Area / pi)
    area = cv2.contourArea(hull)
    diameter = 2 * np.sqrt(area / np.pi)
    
    # 7. วาดภาพ Visualization (เส้นขอบเขียวของ Hull)
    if save_viz_path:
        viz_img = img.copy()
        # วาดเส้น Hull สีเขียวหนา 2px
        cv2.drawContours(viz_img, [hull], -1, (0, 255, 0), 2)
        
        # (Optional) วาด Contour ดิบสีแดงจางๆ เพื่อเปรียบเทียบ
        # cv2.drawContours(viz_img, [best_contour], -1, (0, 0, 255), 1)
        
        cv2.imwrite(save_viz_path, viz_img)

    return diameter

def calculate_refined_baseline(baseline_diameters):
    """คำนวณค่าเฉลี่ย RBC ปกติ โดยใช้ Median เพื่อป้องกันค่ากระโดด (Outliers)"""
    if not baseline_diameters: return 120.0 # ค่าเริ่มต้นกรณีหา Baseline ไม่ได้
    return np.median(baseline_diameters)

def process_folder_sizes(case_folder_path):
    """
    วิเคราะห์ขนาดและรูปร่างเซลล์ในโปรเจกต์ MalariaX
    """
    # โฟลเดอร์เป้าหมายที่จะวิเคราะห์
    TARGET_FOLDERS = ["1chromatin", "band form", "basket form", "schuffner dot", "Appliqué"]
    
    # โฟลเดอร์ที่ใช้เป็นมาตรฐาน (Normal Cell)
    possible_baseline = ["nomal_cell", "normal_cell", "normal"]
    baseline_path = None
    
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
                
                # ใช้ฟังก์ชันใหม่ที่มี Convex Hull
                d = get_diameter_and_visualize(full_p)
                
                # Baseline ต้องกลมและมีขนาดสมเหตุสมผล
                # (เรียกใช้ฟังก์ชันเช็ค Shape จากไฟล์ cellree.py)
                try:
                    circ, _ = cellree.analyze_shape(full_p)
                    if d > 40 and circ > 0.70:
                        baseline_diameters.append(d)
                except:
                    # กรณีไม่มี cellree หรือ error ให้ข้ามไปก่อน
                    if d > 40: baseline_diameters.append(d)
    
    baseline_A = calculate_refined_baseline(baseline_diameters)
    print(f"📊 Baseline A (Normal RBC size): {baseline_A:.2f} px")

    # --- Step 2: วิเคราะห์เชื้อ (B) ---
    results_summary = {} 
    amoeboid_count = 0 

    for folder_name in TARGET_FOLDERS:
        target_path = os.path.join(case_folder_path, folder_name)
        if not os.path.exists(target_path): continue
        
        viz_folder = os.path.join(VIZ_ROOT, folder_name)
        os.makedirs(viz_folder, exist_ok=True)
            
        for file in os.listdir(target_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(target_path, file)
                viz_out = os.path.join(viz_folder, file)
                
                # 1. วัดขนาด (ใช้ Convex Hull แล้ว)
                size_B = get_diameter_and_visualize(full_path, viz_out)
                
                # 2. วิเคราะห์รูปร่าง
                circ, shape_stat = 0, "Unknown"
                try:
                    circ, shape_stat = cellree.analyze_shape(full_path)
                except:
                    pass
                
                if shape_stat == "Amoeboid":
                    amoeboid_count += 1
                    # เขียน Text บนภาพ Viz
                    tmp = cv2.imread(viz_out)
                    if tmp is not None:
                        cv2.putText(tmp, f"Amoeboid ({circ:.2f})", (5, 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.imwrite(viz_out, tmp)

                # 3. คำนวณ Ratio (B/A)
                ratio = size_B / baseline_A if baseline_A > 0 else 0
                
                results_summary[file] = {
                    "folder": folder_name,
                    "size_px": round(size_B, 2),
                    "ratio": round(ratio, 2),
                    "size_status": "Enlarged" if ratio > 1.20 else "Normal",
                    "shape_status": shape_stat,
                    "circularity": round(circ, 4),
                    "viz_image": viz_out 
                }

    return results_summary, amoeboid_count