import cv2
import numpy as np
import math

def calculate_marginal_ratio(image_path, save_viz_path=None):
    """
    ปรับปรุง: เน้นการหาขอบเขตเซลล์ (Segmentation) ให้เนียนขึ้นด้วย Otsu + Convex Hull
    และคำนวณ Marginal Ratio แบบ Radial Projection (เส้นตรง)
    """
    img = cv2.imread(image_path)
    if img is None: return 0.0
    
    # --- 1. Preprocessing (แก้ใหม่เพื่อให้ได้รูปทรงเซลล์ที่ดีขึ้น) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # GaussianBlur ช่วยลด Noise เล็กๆ น้อยๆ
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # [แก้จุดที่ 1] ใช้ Otsu's Thresholding แทน Adaptive
    # Otsu จะหาค่า Threshold กลางที่แยก Background กับ Cell ได้ดีกว่าในภาพ Microscope
    _, cell_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology: Closing เพื่อถมรูเล็กๆ ภายในเซลล์ให้เต็ม
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cell_mask = cv2.morphologyEx(cell_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # หา Contours
    cell_cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cell_cnts: return 0.0
    
    # เลือก Contour ที่ใหญ่ที่สุด (สมมติว่าเป็นเซลล์หลัก)
    raw_cnt = max(cell_cnts, key=cv2.contourArea)
    
    # [แก้จุดที่ 2] ใช้ Convex Hull
    # ช่วยแก้ปัญหาขอบหยักๆ ให้กลายเป็นรูปทรงโค้งมน (เหมือนเม็ดเลือดแดงจริง)
    main_cell_cnt = cv2.convexHull(raw_cnt)
    
    # สร้าง Mask สุดท้ายจาก Hull ที่เรียบเนียนแล้ว
    final_cell_mask = np.zeros_like(gray)
    cv2.drawContours(final_cell_mask, [main_cell_cnt], -1, 255, -1)

    # --- 2. หาจุดศูนย์กลางเซลล์ (Centroid) ---
    Mc = cv2.moments(main_cell_cnt)
    if Mc['m00'] == 0: return 0.0
    cx, cy = int(Mc['m10']/Mc['m00']), int(Mc['m01']/Mc['m00'])

    # --- 3. หาจุดโครมาทิน (Darkest Point) ---
    # ใช้ Mask เพื่อหาจุดมืดสุดเฉพาะในเขตเซลล์
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, mask=final_cell_mask)
    px, py = min_loc

    # --- 4. คำนวณจุดตัดขอบเซลล์ (Radial Projection) ---
    # คำนวณองศาจาก Center ไปหา Chromatin
    target_angle = math.atan2(py - cy, px - cx)
    
    best_edge_point = (px, py)
    min_angle_diff = float('inf')

    # วนลูปจุดบน Convex Hull เพื่อหาจุดที่ตรงกับองศานี้ที่สุด
    # (Convex Hull มีจำนวนจุดน้อยกว่า Contour ดิบ ทำให้ทำงานเร็วและแม่นยำกว่า)
    for point in main_cell_cnt:
        ex, ey = point[0]
        if ex == cx and ey == cy: continue # ป้องกันจุดซ้ำศูนย์กลาง
        
        curr_angle = math.atan2(ey - cy, ex - cx)
        
        # คำนวณความต่างขององศา (จัดการเรื่องวงกลม 360 องศา -PI ถึง PI)
        diff = abs(curr_angle - target_angle)
        if diff > math.pi:
            diff = 2 * math.pi - diff
            
        if diff < min_angle_diff:
            min_angle_diff = diff
            best_edge_point = (ex, ey)

    # --- 5. คำนวณ Ratio ---
    # ระยะจาก Center -> Chromatin
    dist_c_to_p = math.sqrt((px - cx)**2 + (py - cy)**2)
    
    # ระยะจาก Center -> Edge (รัศมีรวมในทิศทางนั้น)
    bx, by = best_edge_point
    dist_c_to_edge = math.sqrt((bx - cx)**2 + (by - cy)**2)
    
    if dist_c_to_edge == 0: 
        ratio = 0.0
    else:
        ratio = dist_c_to_p / dist_c_to_edge

    # --- 6. Visualization ---
    if save_viz_path:
        viz = np.zeros_like(img)
        
        # วาดเส้นขอบ (Convex Hull) สีขาว
        cv2.drawContours(viz, [main_cell_cnt], -1, (255, 255, 255), 1)
        
        # วาดเส้นไกด์ไลน์จางๆ จาก Center -> Edge
        cv2.line(viz, (cx, cy), best_edge_point, (50, 50, 50), 1)

        # เส้น Center -> Chromatin (สีฟ้า)
        cv2.line(viz, (cx, cy), (px, py), (255, 255, 0), 2)
        
        # เส้น Chromatin -> Edge (สีชมพู)
        cv2.line(viz, (px, py), best_edge_point, (255, 0, 255), 2)

        # จุด Marker
        cv2.circle(viz, (cx, cy), 3, (0, 255, 0), -1)      # เขียว (Center)
        cv2.circle(viz, (px, py), 3, (0, 0, 255), -1)      # แดง (Chromatin)
        cv2.circle(viz, best_edge_point, 3, (255, 0, 255), -1) # ชมพู (Edge)

        cv2.imwrite(save_viz_path, viz)

    return round(min(ratio, 1.0), 4)