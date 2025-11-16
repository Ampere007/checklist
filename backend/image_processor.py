# image_processor.py
import cv2
import numpy as np

def detect_combined_edges(cell_image):
    """
    ฟังก์ชันสำหรับทำ Edge Detection
    - รับภาพเซลล์ (BGR) เข้ามา 1 ภาพ
    - คืนค่าเป็นภาพเส้นขอบ (ขาว-ดำ) ที่รวมขอบของ Chromatin และ Cell แล้ว
    """
    if cell_image is None:
        return None

    # === ส่วนที่ 1: หาขอบของ Chromatin ===
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([140, 110, 50])
    upper_purple = np.array([175, 255, 220])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # ใช้ Canny กับ mask จะให้ผลลัพธ์ที่ดีและเร็วกว่า
    chromatin_edges = cv2.Canny(mask_cleaned, 100, 200)

    # === ส่วนที่ 2: หาขอบของเซลล์เม็ดเลือดแดง ===
    gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR_GRAY)
    _, cell_mask = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cell_edges = cv2.Canny(cell_mask, 100, 200)

    # === ส่วนที่ 3: รวมขอบทั้งสองอย่างเข้าด้วยกัน ===
    combined_edges = cv2.bitwise_or(chromatin_edges, cell_edges)

    return combined_edges