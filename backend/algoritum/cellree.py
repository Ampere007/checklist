import cv2
import numpy as np

def analyze_shape(image_path):
    """
    วิเคราะห์รูปร่าง (Morphology Analysis) แบบเน้นโครงสร้างหลักของ RBC
    คืนค่า: (circularity, shape_status)
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0, "Unknown"

    # 1. ปรับภาพให้นุ่มนวลเพื่อลบจุดยึกยักภายใน (Noise Removal)
    # ใช้ Median Blur จะช่วยรักษาขอบนอกแต่ลบรายละเอียดเชื้อภายในได้ดีกว่า
    blurred = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 2. ใช้ Adaptive Threshold ร่วมกับ Canny หรือ Otsu 
    # ในกรณีนี้เราจะใช้ Grayscale Threshold เพื่อให้ได้ขอบ RBC ที่ชัดเจน
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. ใช้ Morphology Closing เพื่อเชื่อมขอบนอกให้สนิทและปิดรูภายใน
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Hole Filling (ถมรูตรงกลางให้เต็ม 100%)
    # เพื่อป้องกันไม่ให้ Perimeter ไปนับรอยหยักข้างใน
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # 5. คำนวณ Circularity จาก Contour ที่ใหญ่ที่สุด
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, "Unknown"

    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)

    if perimeter == 0:
        return 0, "Unknown"

    # Circularity Formula: (4 * pi * Area) / (Perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # ✨ การปรับเกณฑ์: 
    # RBC ที่ติดเชื้อจะเบี้ยวเล็กน้อยอยู่แล้ว (ปกติ 0.75-0.85)
    # ถ้าค่าต่ำกว่า 0.70 คือ Amoeboid ที่แท้จริง
    shape_status = "Amoeboid" if circularity < 0.70 else "Round"

    return circularity, shape_status