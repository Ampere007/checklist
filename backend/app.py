import os
import uuid
import traceback
import shutil
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import Counter
from ultralytics import YOLO 

# --- Import Pipeline ---
from cellpose_segmenter import segment_and_save_cells, filter_bad_cells
from image_processor import preprocess_image_with_mask
from model_loader import load_resnet_model, predict_image_file 

# Import Algorithms
from algoritum.findsize import process_folder_sizes 
from algoritum.diastant import calculate_marginal_ratio 
from algoritum.yolo_counter import count_chromatin_with_yolo
from algoritum import removebg  # <--- เรียกใช้ไฟล์ตัดพื้นหลัง

app = Flask(__name__)
CORS(app)

# ================== SETUP FOLDERS ==================
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented_cells'
PROCESSED_FOLDER = 'processed_results'
DEBUG_FOLDER = 'debug_crops'  # <--- โฟลเดอร์เก็บรูปที่ตัดแล้ว

# สร้างโฟลเดอร์ให้ครบ
for folder in [UPLOAD_FOLDER, SEGMENTED_FOLDER, PROCESSED_FOLDER, DEBUG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ================== LOAD MODELS ==================
print("🚀 Loading System...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load ResNet
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_resnet-50_new_start.pth') 
CLASS_NAMES = ['1chromatin', 'band form', 'basket form', 'nomal_cell', 'schuffner dot'] 
resnet_model, device = load_resnet_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

# 2. Load YOLO
YOLO_PATH = os.path.join(BASE_DIR, 'model', 'best.pt') 
print(f"📦 Loading YOLOv8 from {YOLO_PATH}...")
try:
    yolo_model = YOLO(YOLO_PATH)
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    yolo_model = None

# ================== ROUTES ==================

@app.route('/uploads/<path:filename>')
def send_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/cells/<path:path>')
def send_cell_image(path): 
    return send_from_directory(SEGMENTED_FOLDER, path)

@app.route('/processed/<path:path>')
def send_processed_image(path):
    return send_from_directory(PROCESSED_FOLDER, path)

# Route สำหรับส่งรูปที่ Crop แล้วให้หน้าเว็บ
@app.route('/debug_crops/<path:filename>')
def send_debug_image(filename):
    return send_from_directory(DEBUG_FOLDER, filename)

# ================== MAIN API ==================

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    filepath = None
    try:
        unique_id = str(uuid.uuid4())
        # เก็บชื่อไฟล์ดั้งเดิมไว้
        original_filename = unique_id + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(filepath)

        # ==================================================================================
        # 0️⃣ Step 0: Remove Background / Crop Square
        # ==================================================================================
        print("0️⃣ Preprocessing: Cropping Inner Square...")
        
        # ตั้งค่า Default URL เป็นรูปต้นฉบับก่อน (เผื่อ Crop ไม่ผ่าน)
        final_image_url = f"uploads/{original_filename}"
        
        try:
            # 1. ส่งรูปเข้า Algorithm ตัดให้เหลือแค่สี่เหลี่ยมด้านใน
            cleaned_img_bgr = removebg.process_image(filepath)

            # 2. ตั้งชื่อไฟล์ใหม่ (เติม crop_ ข้างหน้า)
            cleaned_filename = "crop_" + original_filename
            cleaned_filepath = os.path.join(DEBUG_FOLDER, cleaned_filename) # เซฟลง debug_crops

            # 3. บันทึกรูปลง Disk
            cv2.imwrite(cleaned_filepath, cleaned_img_bgr)
            print(f"✅ Image cropped. Saved at: {cleaned_filepath}")

            # 4. [สำคัญมาก] เปลี่ยน filepath ให้ Pipeline ไปใช้รูปที่ตัดแล้วทำงานต่อ
            filepath = cleaned_filepath
            
            # 5. [สำคัญมาก] เปลี่ยน URL ที่จะส่งกลับหน้าเว็บ ให้เป็นรูปที่ตัดแล้ว
            # เพื่อให้พิกัด Bounding Box ตรงกับภาพที่แสดง
            final_image_url = f"debug_crops/{cleaned_filename}"

        except Exception as e:
            print(f"⚠️ Cropping failed (using original image instead): {e}")
            # ถ้า Error ก็ใช้ filepath เดิมทำงานต่อ
        # ==================================================================================
        
        # 1. Segmentation (ทำบนรูปที่ Crop แล้ว)
        print(f"1️⃣ Running Cellpose Segmentation...")
        raw_cells_data = segment_and_save_cells(filepath)
        if not raw_cells_data: return jsonify({'message': 'No cells found.', 'success': False})
        
        # 2. Filtering
        print(f"2️⃣ Filtering cells...")
        valid_cells_data = filter_bad_cells(raw_cells_data)
        if not valid_cells_data: return jsonify({'message': 'All cells filtered.', 'success': False})

        # Prepare folders
        first_cell_path = valid_cells_data[0]['file_path']
        session_id = os.path.basename(os.path.dirname(first_cell_path))
        sorted_base_dir = os.path.join(PROCESSED_FOLDER, session_id, 'sorted_by_morphology')
        
        for class_name in CLASS_NAMES + ['Unknown']:
            os.makedirs(os.path.join(sorted_base_dir, class_name), exist_ok=True)

        analysis_results = []
        counts = Counter()

        # 3. Classification
        print(f"3️⃣ Classifying {len(valid_cells_data)} cells...")
        CONFIDENCE_THRESHOLD = 90.0

        for cell_item in valid_cells_data:
            cell_path = cell_item['file_path']
            bbox = cell_item['bbox']
            cell_filename = os.path.basename(cell_path)
            
            # Preprocess for ResNet
            temp_masked_path = cell_path.replace(".png", "_temp_mask.png")
            try:
                masked_img = preprocess_image_with_mask(cell_path)
                if masked_img: masked_img.save(temp_masked_path)
                else: shutil.copy(cell_path, temp_masked_path)
            except: shutil.copy(cell_path, temp_masked_path)

            # Predict ResNet
            predicted_label = "Unknown"
            confidence = 0.0
            if resnet_model is not None:
                predicted_label, confidence = predict_image_file(resnet_model, device, temp_masked_path)
                if predicted_label != 'nomal_cell' and confidence < CONFIDENCE_THRESHOLD:
                    predicted_label = 'nomal_cell' 

            # --- Chromatin Analysis ---
            marginal_ratio = 0.0
            chromatin_count = 0
            chromatin_bboxes = []
            distance_viz_url = None 
            
            if predicted_label == '1chromatin':
                # B1. วัดระยะห่าง
                try:
                    dist_viz_filename = cell_filename.replace(".png", "_dist_viz.png")
                    dist_viz_path = os.path.join(sorted_base_dir, predicted_label, dist_viz_filename)
                    
                    marginal_ratio = calculate_marginal_ratio(cell_path, save_viz_path=dist_viz_path)
                    
                    distance_viz_url = f"processed/{session_id}/sorted_by_morphology/{predicted_label}/{dist_viz_filename}"
                except Exception as e:
                    print(f"Distance calc error: {e}")

                # B2. นับจำนวน YOLO
                if yolo_model is not None:
                    try:
                        count, bboxes = count_chromatin_with_yolo(yolo_model, cell_path)
                        chromatin_count = count
                        chromatin_bboxes = bboxes 
                        if chromatin_count == 0: chromatin_count = 1
                    except Exception as e:
                        print(f"YOLO error: {e}")
                        chromatin_count = 1
                else:
                    chromatin_count = 1

            # Cleanup & Sort
            if os.path.exists(temp_masked_path):
                try: os.remove(temp_masked_path)
                except: pass

            target_path = os.path.join(sorted_base_dir, predicted_label, cell_filename)
            shutil.copy(cell_path, target_path)

            analysis_results.append({
                "cell": cell_filename,
                "characteristic": predicted_label,
                "confidence": f"{confidence:.2f}%",
                "marginal_ratio": marginal_ratio,
                "chromatin_count": chromatin_count,
                "chromatin_bboxes": chromatin_bboxes,
                "distance_viz_url": distance_viz_url,
                "url": f"cells/{session_id}/{cell_filename}",
                "bbox": bbox
            })
            counts[predicted_label] += 1

        # 4. Size Analysis
        print(f"4️⃣ Analyzing Sizes...")
        size_data_raw, amoeboid_count = process_folder_sizes(sorted_base_dir)
        
        size_analysis_for_web = []
        if size_data_raw:
            for fname, details in size_data_raw.items():
                viz_url = None
                if details.get('viz_image'):
                    rel_path = os.path.relpath(details['viz_image'], PROCESSED_FOLDER).replace("\\", "/")
                    viz_url = f"processed/{rel_path}"

                size_analysis_for_web.append({
                    "filename": fname,
                    "folder": details['folder'],
                    "size_px": details['size_px'],
                    "ratio": details['ratio'],
                    "status": details.get('size_status', 'Unknown'),
                    "shape": details.get('shape_status', 'Unknown'),
                    "circularity": details.get('circularity', 0),
                    "visualization_url": viz_url 
                })

        # Overall Diagnosis
        overall_diagnosis = "Normal / No Parasite Detected"
        if counts['schuffner dot'] > 0: overall_diagnosis = "P. vivax Detected"
        elif counts['band form'] > 0 or counts['basket form'] > 0: overall_diagnosis = "P. malariae Detected"
        elif counts['1chromatin'] > 0: overall_diagnosis = "P. falciparum Detected"
        
        if amoeboid_count > 2 and overall_diagnosis == "Normal / No Parasite Detected":
            overall_diagnosis = "Potential P. vivax (Amoeboid forms observed)"

        return jsonify({
            "session_id": session_id,
            # ส่ง URL ของภาพที่ Crop แล้วกลับไปให้หน้าเว็บแสดงผล (เพื่อให้กรอบแดงตรงตำแหน่ง)
            "original_image_url": final_image_url, 
            "overall_diagnosis": overall_diagnosis,
            "total_cells_segmented": len(valid_cells_data),
            "vit_characteristics": analysis_results, 
            "size_analysis": size_analysis_for_web, 
            "amoeboid_count": amoeboid_count,
            "summary": dict(counts),
            "success": True
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)