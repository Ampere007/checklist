import os
import uuid
import traceback
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import Counter

# --- 1. Import Pipeline ใหม่ ---
from cellpose_segmenter import segment_and_save_cells
from grabcut_processor import process_cells_with_grabcut 
# (FIXED) Import Edge Detector (ที่ตอนนี้อยู่ใน chromatin.py)
from processing.chromatin import create_edge_images
# ---
from processing.SchuffnerFormFinderAlgorithm import process_schuffner
from processing.BasketORBand import process_basket_band
from model_loader import (
    load_classification_model,
    run_prediction,
    get_transform,
    load_yolo_model,
    run_yolo_prediction,
)
from processing.distance_analyzer import process_image as analyze_cell_distances

# SECTION 1: ฟังก์ชันอัลกอริทึม
def count_chromatin_dots(image_path):
    # (ฟังก์ชันนี้ยังอยู่ตามเดิม)
    image = cv2.imread(image_path)
    if image is None: return 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = 120
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area, max_area = 10, 500
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    return len(valid_contours)

def diagnose_by_scoring(features):
    # (ฟังก์ชันนี้ยังอยู่ตามเดิม)
    scores = {"P. falciparum": 0, "P. vivax": 0, "P. malariae": 0}
    if features.get("is_near_edge"): scores["P. falciparum"] += 15; scores["P. vivax"] -= 5; scores["P. malariae"] -= 5
    if features.get("is_band_form"): scores["P. malariae"] += 30; scores["P. falciparum"] -= 25; scores["P. vivax"] -= 25
    if features.get("has_schuffners"): scores["P. vivax"] += 20; scores["P. falciparum"] -= 25; scores["P. malariae"] -= 10
    if features.get("chromatin_count", 0) > 1: scores["P. falciparum"] += 20; scores["P. vivax"] -= 15; scores["P. malariae"] -= 15
    diagnosis = "Normal/Undetermined"
    if max(scores.values()) > 0: diagnosis = max(scores, key=scores.get)
    return diagnosis, scores

# SECTION 2: การตั้งค่า Flask App และโหลดโมเดล
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented_cells'
PROCESSED_CELLS_FOLDER = 'grabcut_processed_cells' 
PROCESSED_FOLDER = 'processed_results'
for folder in [UPLOAD_FOLDER, SEGMENTED_FOLDER, PROCESSED_CELLS_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

print("Loading AI models...")
SCHUFFNER_MODEL_PATH = 'models/Schuffner.pt'
BASKET_MODEL_PATH = 'models/bastket_band.pt'
CHROMATIN_MODEL_PATH = 'models/Chromatindetect.pt'
CLASS_NAMES_RESNET = ['not_found', 'found']
schuffner_model = load_classification_model(SCHUFFNER_MODEL_PATH, num_classes=len(CLASS_NAMES_RESNET))
basket_model = load_classification_model(BASKET_MODEL_PATH, num_classes=len(CLASS_NAMES_RESNET))
image_transform_resnet = get_transform()
chromatin_model = load_yolo_model(CHROMATIN_MODEL_PATH) 
print("All AI models loaded successfully.")

# SECTION 3: (FIXED) เพิ่ม Route ใหม่สำหรับ YOLO
@app.route('/processed_cells/<path:path>')
def send_processed_cell_image(path):
    return send_from_directory(PROCESSED_CELLS_FOLDER, path)
@app.route('/results/<path:path>')
def send_result_image(path):
    return send_from_directory(PROCESSED_FOLDER, path)
@app.route('/cells/<path:path>')
def send_cell_image(path):
    return send_from_directory(SEGMENTED_FOLDER, path)

# (FIXED) เพิ่ม Route นี้สำหรับเสิร์ฟ "ภาพที่มีกรอบ"
@app.route('/yolo_results/<path:path>')
def send_yolo_image(path):
    # Path นี้จะชี้ไปที่ processed_results/<session_id>/yolo_results
    return send_from_directory(PROCESSED_FOLDER, path)

# SECTION 4: (FIXED) อัปเดต API Endpoint หลัก
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file'];
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    filepath = None
    try:
        unique_id = str(uuid.uuid4()); file_extension = os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, unique_id + file_extension); file.save(filepath)
        
        # --- ขั้นตอนที่ 1: Cellpose (Crop) ---
        cropped_cell_paths = segment_and_save_cells(filepath)
        if not cropped_cell_paths: return jsonify({'message': 'No cells were found in the image.'})
        
        # --- ขั้นตอนที่ 2: GrabCut (ลบพื้นหลัง) ---
        print(f"Running GrabCut on {len(cropped_cell_paths)} cells...")
        final_processed_paths = process_cells_with_grabcut(cropped_cell_paths)
        if not final_processed_paths: return jsonify({'error': 'Cell processing (GrabCut) failed.'}), 500
        
        # --- ขั้นตอนที่ 3: Edge Detection ---
        processing_input_dir = os.path.dirname(final_processed_paths[0]) # (ภาพสีที่ลบพื้นหลัง)
        session_id = os.path.basename(processing_input_dir)
        edge_output_dir = os.path.join(PROCESSED_FOLDER, session_id, 'edge_detect_results')
        
        print(f"Running Edge Detection for YOLO...")
        edge_image_paths, edge_output_dir = create_edge_images(processing_input_dir, edge_output_dir)
        if not edge_image_paths: return jsonify({'error': 'Edge detection failed.'}), 500
        
        # --- ตั้งค่า Paths อื่นๆ ---
        segmented_input_dir = os.path.dirname(cropped_cell_paths[0]) # (ภาพ Crop ดั้งเดิม)
        schuffner_output_dir = os.path.join(PROCESSED_FOLDER, session_id, 'schuffner')
        basket_band_output_dir = os.path.join(PROCESSED_FOLDER, session_id, 'basket_band')
        distance_analysis_output_dir = os.path.join(PROCESSED_FOLDER, session_id, 'distance_analysis_results')
        # (FIXED) โฟลเดอร์ใหม่สำหรับเก็บผลลัพธ์ YOLO
        yolo_output_dir = os.path.join(PROCESSED_FOLDER, session_id, 'yolo_results')
        
        # (FIXED) เพิ่ม yolo_output_dir
        for d in [schuffner_output_dir, basket_band_output_dir, distance_analysis_output_dir, yolo_output_dir]:
            os.makedirs(d, exist_ok=True)

        # --- รัน Pre-processing (บนภาพ GrabCut) ---
        print("Running pre-processing (Schuffner, Basket)...")
        process_schuffner(processing_input_dir, schuffner_output_dir)
        process_basket_band(processing_input_dir, basket_band_output_dir)

        # --- รันโมเดล AI (รอบที่ 1) ---
        print("Running AI models (Schuffner, Basket, YOLO)...")
        schuffner_summary = run_prediction(schuffner_model, schuffner_output_dir, image_transform_resnet, CLASS_NAMES_RESNET)
        basket_summary = run_prediction(basket_model, basket_band_output_dir, image_transform_resnet, CLASS_NAMES_RESNET)
        
        # (FIXED) เรียก YOLO ด้วย 4 arguments
        chromatin_summary = run_yolo_prediction(
            chromatin_model,
            edge_output_dir,        # 1. ที่อยู่ "ภาพขอบ" (สำหรับ Predict)
            processing_input_dir,   # 2. ที่อยู่ "ภาพสี" (สำหรับวาด)
            yolo_output_dir         # 3. ที่อยู่ "ผลลัพธ์" (สำหรับบันทึก)
        )
        
        # (FIXED) ตรรกะนี้ยังจำเป็นสำหรับ Schuffner/Basket
        def get_abnormal_cell_info(summary, session_id, prefix):
            info = []
            for path in summary.get("found_paths", []):
                filename = os.path.basename(path)
                if prefix: filename = filename.replace(prefix, '')
                filename = filename.replace('_processed', '')
                # (FIXED) URL นี้ถูกต้องสำหรับ Schuffner/Basket (ชี้ไปที่ภาพ crop ดั้งเดิม)
                url_path = os.path.join("cells", session_id, filename).replace(os.sep, '/')
                info.append({"file": filename, "url": url_path})
            return info
        
        schuffner_summary["abnormal_cells"] = get_abnormal_cell_info(schuffner_summary, session_id, 'schuffner_')
        basket_summary["abnormal_cells"] = get_abnormal_cell_info(basket_summary, session_id, 'basket_band_')
        
        # (FIXED) ตรรกะใหม่สำหรับสร้าง "abnormal_cells" ของ Chromatin
        chromatin_abnormal_cells = []
        for yolo_result_path in chromatin_summary.get("found_paths", []):
            # yolo_result_path คือ '.../yolo_results/yolo_cell_crop_1_processed.png'
            yolo_filename = os.path.basename(yolo_result_path) # 'yolo_cell_crop_1_processed.png'
            
            # 1. ชื่อไฟล์ดั้งเดิม (สำหรับ Scoring)
            original_filename = yolo_filename.replace('yolo_', '').replace('_processed', '') # 'cell_crop_1.png'
            
            # 2. URL ที่ Frontend จะใช้ (ชี้ไปที่ภาพที่มีกรอบ)
            # 'fea.../yolo_results/yolo_cell_crop_1_processed.png'
            # (FIXED) URL ต้องชี้ไปที่ Route '/yolo_results/' ไม่ใช่ '/cells/'
            url_path = os.path.join("yolo_results", session_id, 'yolo_results', yolo_filename).replace(os.sep, '/')
            
            chromatin_abnormal_cells.append({
                "file": original_filename, # ใช้ชื่อเดิมสำหรับอ้างอิง
                "url": url_path             # ใช้ URL ใหม่สำหรับแสดงผล
            })
        chromatin_summary["abnormal_cells"] = chromatin_abnormal_cells
        # ---

        # --- สร้าง Set (ตรรกะนี้จะทำงานถูกต้อง) ---
        found_schuffner_files = {info['file'] for info in schuffner_summary.get("abnormal_cells", [])}
        found_basket_files = {info['file'] for info in basket_summary.get("abnormal_cells", [])}
        found_chromatin_cells = {info['file']for info in chromatin_summary.get("abnormal_cells", [])} # ได้ 'cell_crop_1.png'

        distance_analysis_results = []; cell_distances = {}
        
        # --- วนลูป Distance Analysis (เหมือนเดิม) ---
        print("Running distance analysis...")
        for cell_filename in found_chromatin_cells: # ใช้ชื่อ 'cell_crop_1.png'
            original_cell_path = os.path.join(segmented_input_dir, cell_filename) # ถูกต้อง
            if os.path.exists(original_cell_path):
                image_to_process = cv2.imread(original_cell_path);
                if image_to_process is None: continue
                result_image, distance_data_list = analyze_cell_distances(image_to_process)
                result_image_filename = f"dist_{cell_filename}"
                result_image_path = os.path.join(distance_analysis_output_dir, result_image_filename)
                cv2.imwrite(result_image_path, result_image)
                if distance_data_list:
                    distances = distance_data_list[0]
                    min_dist = min((d[0] for d in distances.values() if isinstance(d, tuple) and d[0] >= 0), default=-1)
                    cell_distances[cell_filename] = min_dist
                    diagnosis = "อาจเป็น P. falciparum (Ring form)" if 0 <= min_dist <= 20 else "ลักษณะโครมาตินปกติ"
                    result_url = os.path.join(session_id, 'distance_analysis_results', result_image_filename).replace(os.sep, '/')
                    distance_analysis_results.append({"cell_file": cell_filename, "min_distance_px": round(min_dist, 2), "diagnosis": diagnosis, "result_url": result_url})

        # --- วนลูป Scoring (เหมือนเดิม) ---
        print("Scoring all cells...")
        total_scores = Counter()
        for cell_path in cropped_cell_paths:
            cell_filename = os.path.basename(cell_path)
            has_schuffners_feature = cell_filename in found_schuffner_files
            is_band_form_feature = cell_filename in found_basket_files
            min_dist = cell_distances.get(cell_filename, -1); is_near_edge = 0 <= min_dist <= 20
            chromatin_count = count_chromatin_dots(cell_path) 
            features = {"has_schuffners": has_schuffners_feature, "is_band_form": is_band_form_feature, "chromatin_count": chromatin_count, "is_near_edge": is_near_edge}
            _ , scores = diagnose_by_scoring(features); total_scores.update(scores)

        # --- สรุปผล (เหมือนเดิม) ---
        if not any(score > 0 for score in total_scores.values()): overall_diagnosis = "ไม่พบเชื้อมาลาเรีย (No Malaria Parasites Found)"
        else: most_common_species = max(total_scores, key=total_scores.get); overall_diagnosis = f"เชื้อที่พบมากที่สุดคือ: {most_common_species}"
        
        diagnosis_scores = dict(total_scores)
        
        # (FIXED) อัปเดตการนับจำนวน 'found' ให้ถูกต้องตาม summary ที่ได้จาก model_loader
        schuffner_summary["correct_found_count"] = schuffner_summary.get("predictions", {}).get("found", 0)
        basket_summary["correct_found_count"] = basket_summary.get("predictions", {}).get("found", 0)
        chromatin_summary["correct_found_count"] = chromatin_summary.get("predictions", {}).get("found", 0)


        print("Analysis complete. Returning response.")
        return jsonify({
            "overall_diagnosis": overall_diagnosis, "diagnosis_scores": diagnosis_scores,
            "message": "AI analysis complete!", "total_cells_segmented": len(cropped_cell_paths),
            "schuffner_prediction": schuffner_summary, "basket_band_prediction": basket_summary,
            "chromatin_prediction": chromatin_summary, "distance_analysis": distance_analysis_results
        })
    except Exception as e:
        print(f"An error occurred: {e}"); traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath): os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5001)