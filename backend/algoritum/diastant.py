import cv2
import numpy as np
import math

def calculate_marginal_ratio(image_path, save_viz_path=None):
    """
    คำนวณระยะห่างโครมาทินกับขอบเซลล์ (Marginal Ratio)
    และสร้างภาพ Visualization เพื่ออธิบาย Algorithm
    """
    img = cv2.imread(image_path)
    if img is None: return 0.0
    
    # 1. Preprocessing: หาขอบเขตของเม็ดเลือดแดง (RBC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ใช้ GaussianBlur ลด Noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # ใช้ Adaptive Threshold จับขอบ RBC
    cell_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 2)
    
    # เชื่อมเส้นขอบ
    kernel = np.ones((5,5), np.uint8)
    cell_mask = cv2.morphologyEx(cell_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # หา Contour
    cell_cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cell_cnts: return 0.0
    
    # เลือกเซลล์ที่ใหญ่ที่สุด
    main_cell_cnt = max(cell_cnts, key=cv2.contourArea)
    
    # สร้าง Mask ของตัวเซลล์
    final_cell_mask = np.zeros_like(gray)
    cv2.drawContours(final_cell_mask, [main_cell_cnt], -1, 255, -1)

    # 2. หาจุดศูนย์กลางเซลล์ (Cell Center: C)
    Mc = cv2.moments(main_cell_cnt)
    if Mc['m00'] == 0: return 0.0
    cx, cy = int(Mc['m10']/Mc['m00']), int(Mc['m01']/Mc['m00'])

    # 3. Finding Chromatin: หาจุดสีเข้มที่สุด (Darkest Point)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, mask=final_cell_mask)
    px, py = min_loc  # พิกัดโครมาทิน

    # 4. คำนวณระยะทาง
    # ระยะ A: จากศูนย์กลาง -> โครมาทิน
    dist_center_to_chromatin = math.sqrt((px - cx)**2 + (py - cy)**2)
    
    # ระยะ B: จากโครมาทิน -> ขอบเซลล์ที่ใกล้ที่สุด
    dist_to_edge = cv2.pointPolygonTest(main_cell_cnt, (px, py), True)
    if dist_to_edge < 0: dist_to_edge = 0
    
    # 5. คำนวณ Ratio
    total_radius = dist_center_to_chromatin + dist_to_edge
    if total_radius == 0: return 0.0
    
    ratio = dist_center_to_chromatin / total_radius
    
    # ✨ 6. ส่วนสร้างภาพ Visualization (พื้นหลังดำ + เส้นกราฟิก)
    if save_viz_path:
        # สร้างภาพพื้นหลังสีดำขนาดเท่าภาพเดิม
        viz = np.zeros_like(img)
        
        # วาดเส้นขอบเซลล์ (สีขาว)
        cv2.drawContours(viz, [main_cell_cnt], -1, (255, 255, 255), 1)
        
        # วาดจุดศูนย์กลาง (สีเขียว)
        cv2.circle(viz, (cx, cy), 3, (0, 255, 0), -1) 
        
        # วาดจุดโครมาทิน (สีแดง)
        cv2.circle(viz, (px, py), 3, (0, 0, 255), -1) 

        # วาดเส้น: ศูนย์กลาง -> โครมาทิน (สีฟ้า Cyan)
        cv2.line(viz, (cx, cy), (px, py), (255, 255, 0), 2)

        # วาดเส้น: โครมาทิน -> ขอบ (สีชมพู Magenta)
        # ต้องหาพิกัดบนขอบที่ใกล้ที่สุดเพื่อลากเส้นไปหา
        min_dist_calc = float('inf')
        closest_edge_point = (px, py)
        
        # วนลูปหาจุดบน Contour ที่ใกล้ (px, py) ที่สุด
        for point in main_cell_cnt:
            pt = tuple(point[0])
            d = math.sqrt((px - pt[0])**2 + (py - pt[1])**2)
            if d < min_dist_calc:
                min_dist_calc = d
                closest_edge_point = pt
        
        # ลากเส้นสีชมพู
        cv2.line(viz, (px, py), closest_edge_point, (255, 0, 255), 2)

        # บันทึกภาพ
        cv2.imwrite(save_viz_path, viz)

    # ปัดเศษและจำกัดค่าไม่ให้เกิน 1.0
    return round(min(ratio, 1.0), 4)