import cv2
import numpy as np
import os

def detect_circle(img_bgr):
    """
    ตรวจจับวงกลมที่ใหญ่ที่สุดในภาพ (Viewport ของกล้องจุลทรรศน์)
    Return: (cx, cy, r)
    """
    h, w = img_bgr.shape[:2]
    
    # แปลงเป็น Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Blur เพื่อลด noise (ช่วยให้ HoughCircles แม่นขึ้น)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # ใช้ HoughCircles หาวงกลม
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.5,
        param1=100,
        param2=30,
        minRadius=int(min(h, w) * 0.3), # ปรับช่วงรัศมีให้กว้างขึ้นเล็กน้อย
        maxRadius=int(min(h, w) * 0.55),
    )

    if circles is None:
        print("Warning: No circle detected, using center crop default.")
        # ถ้าหาไม่เจอ ให้คืนค่าตรงกลางภาพไปเลย
        return w // 2, h // 2, int(min(h, w) * 0.45)

    # เอาวงกลมที่มั่นใจที่สุด (เรียงตามลำดับที่ Algorithm ส่งมา)
    circles = np.uint16(np.around(circles))
    cx, cy, r = circles[0][0] 
    return int(cx), int(cy), int(r)

def crop_inner_square(img_bgr, cx, cy, r):
    """
    ตัดภาพเป็นสี่เหลี่ยมจัตุรัสที่ 'อยู่ภายใน' วงกลม (Inscribed Square)
    เพื่อกำจัดขอบดำ/ขาว ออกไปให้หมด เหลือแต่เนื้อเซลล์
    """
    h, w = img_bgr.shape[:2]

    # คำนวณระยะจากจุดศูนย์กลางไปยังขอบสี่เหลี่ยมด้านใน
    # จากสูตร Pythagoras: r^2 = x^2 + x^2  =>  r = x * sqrt(2)  => x = r / sqrt(2)
    # เราคูณ 0.70 (ประมาณ 1/sqrt(2)) เพื่อความปลอดภัยไม่ให้ติดขอบดำ
    half_side = int(r * 0.70) 

    # คำนวณพิกัด (Left, Top, Right, Bottom)
    x1 = cx - half_side
    y1 = cy - half_side
    x2 = cx + half_side
    y2 = cy + half_side

    # ตรวจสอบขอบเขต (Clamp) ไม่ให้หลุดเฟรมภาพจริง
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # ตรวจสอบว่าพื้นที่ที่ตัดมาถูกต้องไหม (เผื่อวงกลมหลุดขอบ)
    if x2 <= x1 or y2 <= y1:
        return img_bgr # คืนภาพเดิมถ้าตัดไม่ได้

    # ตัดภาพ (Crop)
    cropped_img = img_bgr[y1:y2, x1:x2].copy()

    return cropped_img

def process_image(image_input):
    """
    Main Function: รับภาพ -> หาวงกลม -> ตัดสี่เหลี่ยมเนื้อใน -> ส่งคืน
    """
    # 1. Load Image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("Input must be a file path or numpy array")

    if img is None:
        raise ValueError("Failed to load image.")

    # 2. Detect Circle
    cx, cy, r = detect_circle(img)

    # 3. Crop Inner Square (ตัดเอาเฉพาะสี่เหลี่ยมข้างใน)
    result_img = crop_inner_square(img, cx, cy, r)

    return result_img

# --- Test Block (รันไฟล์นี้เพื่อทดสอบได้เลย) ---
if __name__ == "__main__":
    test_path = "test.jpg" # เปลี่ยนเป็นชื่อไฟล์รูปของคุณเพื่อเทส
    if os.path.exists(test_path):
        res = process_image(test_path)
        cv2.imwrite("test_cropped_square.jpg", res)
        print("Saved test_cropped_square.jpg")
    else:
        print("No test image found.")