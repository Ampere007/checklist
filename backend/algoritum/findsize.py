import cv2
import math
import numpy as np
import os

def get_diameter_and_visualize(image_path, save_viz_path=None):
    """
    วัดขนาด (Diameter) และคำนวณความกลม (Circularity)
    Return: (diameter, circularity)
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0, 0

    # 1. แปลงเป็น Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. ใช้ Otsu Thresholding แยกเซลล์
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. กำจัด Noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. หาเส้นขอบ (Contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if save_viz_path: cv2.imwrite(save_viz_path, img)
        return 0, 0

    # 5. เลือก Contour ที่ใหญ่ที่สุด
    main_cell_contour = max(contours, key=cv2.contourArea)
    
    # --- คำนวณความกลม (Circularity) เพื่อคัดกรอง ---
    area = cv2.contourArea(main_cell_contour)
    perimeter = cv2.arcLength(main_cell_contour, True)
    
    circularity = 0
    if perimeter > 0:
        # สูตร Circularity: 4*pi*Area / Perimeter^2 (1.0 = กลมดิ๊ก)
        circularity = (4 * np.pi * area) / (perimeter ** 2)

    # --- คำนวณขนาดด้วย MinEnclosingCircle (สีเหลือง) ---
    ((x, y), radius) = cv2.minEnclosingCircle(main_cell_contour)
    diameter = radius * 2
    
    # --- ส่วนวาดภาพ (Visualization) ---
    if save_viz_path:
        viz_img = img.copy()
        
        # วาดเส้นขอบจริง (สีเขียว)
        cv2.drawContours(viz_img, [main_cell_contour], -1, (0, 255, 0), 2)
        
        # วาดวงกลมคำนวณขนาด (สีเหลือง)
        center = (int(x), int(y))
        r = int(radius)
        cv2.circle(viz_img, center, r, (0, 255, 255), 2)
        
        # บันทึกภาพ
        cv2.imwrite(save_viz_path, viz_img)

    return diameter, circularity

def calculate_refined_baseline(baseline_diameters):
    """
    ใช้สถิติ IQR ตัดค่าที่โดดผิดปกติ (Outliers) ออกอีกรอบเพื่อความชัวร์
    """
    if not baseline_diameters:
        return 336.0 # Default

    data = np.array(baseline_diameters)
    
    # หา Q1, Q3 เพื่อดูการกระจายตัวของข้อมูล
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    print(f"   [Stat Log] IQR Filter: เดิม {len(data)} -> เหลือ {len(filtered_data)} ภาพ")
    
    if len(filtered_data) == 0:
        return np.median(data) 
        
    return np.median(filtered_data)

def process_folder_sizes(case_folder_path):
    """
    ฟังก์ชันหลัก
    """
    BASELINE_FOLDER = "nomal_cell"
    TARGET_FOLDERS = ["1chromatin", "band form", "basket form", "schuffner dot"]
    
    VIZ_ROOT_FOLDER = os.path.join(case_folder_path, "size_visualization")
    os.makedirs(VIZ_ROOT_FOLDER, exist_ok=True)
    
    # --- Step 1: หาค่า Baseline (จากเซลล์ปกติที่ "กลม" เท่านั้น) ---
    baseline_path = os.path.join(case_folder_path, BASELINE_FOLDER)
    baseline_diameters = []
    
    viz_baseline_path = os.path.join(VIZ_ROOT_FOLDER, BASELINE_FOLDER)
    os.makedirs(viz_baseline_path, exist_ok=True)

    print(f"\n--- [Step 1] Measuring Normal Cells (Filter: Circularity > 0.75) ---")

    if os.path.exists(baseline_path):
        for file in os.listdir(baseline_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_file = os.path.join(baseline_path, file)
                output_viz_file = os.path.join(viz_baseline_path, file)
                
                # รับค่าทั้งขนาด และ ความกลม
                d, circ = get_diameter_and_visualize(input_file, output_viz_file)
                
                if d > 0: 
                    # ✨ กฎเหล็ก: ต้องกลมเกิน 0.75 ถึงจะยอมรับว่าเป็น Normal Cell ที่ดี ✨
                    # (ช่วยกันพวกเซลล์ซ้อนทับ หรือภาพเบลอๆ ออกไป)
                    if circ > 0.75:
                        baseline_diameters.append(d)
                        print(f"   ✅ {file:<20} : {d:.2f} px (Circ: {circ:.2f})")
                    else:
                        print(f"   ❌ {file:<20} : {d:.2f} px (Circ: {circ:.2f}) -> ตัดทิ้ง (ไม่กลม/ซ้อนทับ)")
    
    # คำนวณ Baseline สุดท้าย
    if baseline_diameters:
        baseline_A = calculate_refined_baseline(baseline_diameters)
        print(f"✅ Final Baseline Size (A) = {baseline_A:.2f} pixels\n")
    else:
        baseline_A = 336.0
        print(f"⚠️ Warning: No valid normal cells found. Using Default A = {baseline_A}\n")

    # --- Step 2: ตรวจเซลล์ผิดปกติ (Target Folders) ---
    results_summary = {} 

    for folder_name in TARGET_FOLDERS:
        target_path = os.path.join(case_folder_path, folder_name)
        viz_target_path = os.path.join(VIZ_ROOT_FOLDER, folder_name)
        
        if not os.path.exists(target_path):
            continue

        os.makedirs(viz_target_path, exist_ok=True)
            
        for file in os.listdir(target_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(target_path, file)
                viz_output_path = os.path.join(viz_target_path, file)
                
                # ตอนตรวจเชื้อโรค ไม่ต้องสนใจความกลม (รับค่า _ ทิ้งไป)
                size_B, _ = get_diameter_and_visualize(full_path, viz_output_path)
                
                ratio_c = 0
                status = "Unknown"
                if size_B > 0:
                    ratio_c = size_B / baseline_A
                    
                    # เกณฑ์: > 1.2 เท่า ถือว่าเริ่มบวม (Enlarged)
                    is_enlarged = ratio_c > 1.2
                    status = "Enlarged" if is_enlarged else "Normal Size"
                
                results_summary[file] = {
                    "folder": folder_name,
                    "size_px": round(size_B, 2),
                    "ratio": round(ratio_c, 2),
                    "status": status,
                    "viz_image": viz_output_path 
                }

    return results_summary

if __name__ == "__main__":
    pass