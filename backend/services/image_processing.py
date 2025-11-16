# backend/services/image_processing.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_chromatin_dots(image_path, show_plot=False):
    """
    ฟังก์ชันสำหรับนับจำนวน Chromatin dots จากภาพของเซลล์เม็ดเลือด
    
    Args:
        image_path (str): ที่อยู่ของไฟล์ภาพ (เช่น 'uploads/cell_image.jpg')
        show_plot (bool): หากเป็น True จะแสดงกราฟผลลัพธ์ (สำหรับดีบัก)

    Returns:
        int: จำนวน Chromatin dots ที่นับได้
    """
    # --- Step 1: โหลดและเตรียมภาพ ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: ไม่สามารถโหลดภาพจาก '{image_path}' ได้")
        return 0

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- Step 2: ทำ Segmentation เพื่อแยก Chromatin ออกมา ---
    # ค่าเหล่านี้อาจจะต้องปรับจูนให้เหมาะกับภาพของคุณ
    threshold_value = 120
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # --- Step 3: ค้นหาและนับ Contours (ก้อนสีขาว) ---
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- Step 4: กรองผลลัพธ์เพื่อความแม่นยำ ---
    # ค่าเหล่านี้อาจจะต้องปรับจูนให้เหมาะกับขนาดโครมาทินในภาพของคุณ
    min_area = 10  # พื้นที่ขั้นต่ำ (กัน noise)
    max_area = 500 # พื้นที่สูงสุด (กันวัตถุขนาดใหญ่)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            
    chromatin_count = len(valid_contours)

    # --- Step 5 (Optional): แสดงผลลัพธ์เพื่อตรวจสอบ ---
    if show_plot:
        output_image = image.copy()
        cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 2)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Binary Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Dots: {chromatin_count}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return chromatin_count

# --- ส่วนนี้สำหรับการทดสอบไฟล์นี้โดยตรง ---
if __name__ == '__main__':
    # ให้สร้างไฟล์ภาพชื่อ 'test_image.jpg' แล้วเอาไปวางไว้ในโฟลเดอร์ backend
    # เพื่อใช้ทดสอบการทำงานของโค้ด
    
    # *** แก้ชื่อไฟล์ตรงนี้ให้เป็นชื่อไฟล์ภาพของคุณ ***
    test_image_file = "test_image.jpg" 
    
    print(f"กำลังทดสอบไฟล์: {test_image_file}")
    num_dots = count_chromatin_dots(test_image_file, show_plot=True)
    
    if num_dots is not None:
        print(f"\nผลการทดสอบ:")
        print(f"จำนวน Chromatin ที่นับได้: {num_dots}")