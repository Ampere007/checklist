import cv2
import numpy as np
from PIL import Image

def preprocess_image_with_mask(image_path):
    """
    ฟังก์ชันสำหรับอ่านรูปภาพและทำ Circular Masking (วงกลมดำ)
    ✨ ปรับปรุงใหม่: ใช้รัศมี 90% ของด้านที่ยาวที่สุด เพื่อให้กระชับขึ้น
       แต่ยังคงครอบคลุมลักษณะสำคัญของเซลล์ได้ดี
    """
    # 1. อ่านรูปด้วย OpenCV
    img = cv2.imread(image_path)
    
    # กรณีไฟล์เสียหรืออ่านไม่ได้ ให้คืนค่าเดิมผ่าน PIL
    if img is None:
        try:
            return Image.open(image_path).convert('RGB')
        except:
            return None

    h, w, c = img.shape
    
    # 2. สร้างหน้ากากวงกลม (Circular Mask)
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w/2), int(h/2))
    
    # ✨ [แก้ไขจุดสำคัญ] ✨
    # ใช้ด้านที่ยาวที่สุดเป็นเกณฑ์ (max(h, w)) แล้วคูณด้วย 0.9 (90%)
    # เพื่อให้วงกลมเล็กลงกว่าขอบรูปเล็กน้อย แต่ยังกว้างพอสำหรับเซลล์ส่วนใหญ่
    radius = int((max(h, w) / 2) * 0.9) 
    
    cv2.circle(mask, center, radius, (255), thickness=-1)

    # 3. ตัดพื้นหลัง (Bitwise AND)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    # 4. แปลงสีจาก BGR เป็น RGB
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    
    # 5. แปลงเป็น PIL Image
    return Image.fromarray(masked_img)