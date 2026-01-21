import os
import uuid
import traceback
import shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import Counter

# --- Import Pipeline ---
from cellpose_segmenter import segment_and_save_cells, filter_bad_cells
from image_processor import preprocess_image_with_mask
from model_loader import load_resnet_model, predict_image_file 

# ✨ Import Algorithm วัดขนาด
from algoritum.findsize import process_folder_sizes

app = Flask(__name__)
CORS(app)

# Setup Folders
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented_cells'
PROCESSED_FOLDER = 'processed_results'

for folder in [UPLOAD_FOLDER, SEGMENTED_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- 1. Load ResNet-50 Model ---
print("🚀 Loading System...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_resnet-50_new_start.pth')
CLASS_NAMES = ['1chromatin', 'band form', 'basket form', 'nomal_cell', 'schuffner dot'] 

model, device = load_resnet_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

# --- Routes สำหรับรูปภาพ ---

@app.route('/cells/<path:path>')
def send_cell_image(path): 
    return send_from_directory(SEGMENTED_FOLDER, path)

@app.route('/processed/<path:path>')
def send_processed_image(path):
    return send_from_directory(PROCESSED_FOLDER, path)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    
    filepath = None
    try:
        unique_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, unique_id + os.path.splitext(file.filename)[1])
        file.save(filepath)
        
        # --- STEP 1: Segment ---
        print(f"1️⃣ Running Cellpose Segmentation...")
        raw_cell_paths = segment_and_save_cells(filepath)
        
        if not raw_cell_paths: 
            return jsonify({'message': 'Cellpose found no cells.'})
        
        # --- STEP 2: Filter ---
        print(f"2️⃣ Filtering cells...")
        cropped_cell_paths = filter_bad_cells(raw_cell_paths)
        
        if not cropped_cell_paths:
             return jsonify({'message': 'All cells were filtered out.'})

        # --- STEP 3: Masking (❌ ปิดส่วนนี้ เพื่อรักษาภาพ Original ไว้) ---
        # print("🎨 Applying circular mask...")
        # for cell_path in cropped_cell_paths:
        #     masked_img = preprocess_image_with_mask(cell_path)
        #     if masked_img:
        #         masked_img.save(cell_path)

        # Prepare Folders for Classification
        input_dir = os.path.dirname(cropped_cell_paths[0])
        session_id = os.path.basename(input_dir)
        
        # กำหนด path ที่จะเก็บผลลัพธ์แยกโฟลเดอร์
        sorted_base_dir = os.path.join(PROCESSED_FOLDER, session_id, 'sorted_by_morphology')
        
        for class_name in CLASS_NAMES + ['Unknown']:
            os.makedirs(os.path.join(sorted_base_dir, class_name), exist_ok=True)

        analysis_results = []
        counts = Counter()

        # --- STEP 4: Classification (✨ แก้ไข: ใช้ Temp Mask สำหรับ AI) ---
        print(f"3️⃣ Classifying {len(cropped_cell_paths)} cells...")
        CONFIDENCE_THRESHOLD = 95.0

        for cell_path in cropped_cell_paths:
            cell_filename = os.path.basename(cell_path)
            predicted_label = "Unknown"
            confidence = 0.0
            
            # 🟢 1. สร้างไฟล์ Mask ชั่วคราว (_temp) เพื่อส่งให้ AI
            temp_masked_path = cell_path.replace(".png", "_temp_mask.png")
            try:
                masked_img = preprocess_image_with_mask(cell_path)
                if masked_img:
                    masked_img.save(temp_masked_path)
                else:
                    # กรณี Mask ไม่ติด ให้ copy ไฟล์เดิมไปแปะแทน (กัน Error)
                    shutil.copy(cell_path, temp_masked_path)
            except Exception:
                 shutil.copy(cell_path, temp_masked_path)

            # 🟢 2. ให้ AI ทำนายจากไฟล์ Temp (ที่มีพื้นหลังดำ)
            if model is not None:
                try:
                    predicted_label, confidence = predict_image_file(model, device, temp_masked_path)
                    
                    if predicted_label != 'nomal_cell' and confidence < CONFIDENCE_THRESHOLD:
                        print(f"⚠️ Low confidence ({confidence:.2f}%) for {predicted_label} -> Normal")
                        predicted_label = 'nomal_cell' 

                except Exception as e:
                    print(f"Error predicting: {e}")
            
            # 🟢 3. ลบไฟล์ Temp ทิ้ง (เสร็จภารกิจ AI แล้ว)
            if os.path.exists(temp_masked_path):
                try:
                    os.remove(temp_masked_path)
                except:
                    pass

            # 🟢 4. Copy ไฟล์ "ต้นฉบับ" (Original) ไปยังโฟลเดอร์ผลลัพธ์
            # เพื่อให้ findsize.py ได้รูปที่มีพื้นหลังสว่างไปคำนวณ
            target_path = os.path.join(sorted_base_dir, predicted_label, cell_filename)
            shutil.copy(cell_path, target_path)

            analysis_results.append({
                "cell": cell_filename,
                "characteristic": predicted_label,
                "confidence": f"{confidence:.2f}%", 
                "url": f"cells/{session_id}/{cell_filename}" 
            })
            counts[predicted_label] += 1

        # --- STEP 5: Size Analysis & Visualization ---
        print(f"4️⃣ Analyzing Cell Sizes & Generating Visualization...")
        
        # ฟังก์ชันนี้จะเข้าไปอ่านไฟล์ใน sorted_base_dir (ซึ่งตอนนี้เป็นภาพ Original แล้ว)
        size_data_raw = process_folder_sizes(sorted_base_dir)
        
        size_analysis_for_web = []
        
        if size_data_raw:
            for filename, details in size_data_raw.items():
                full_viz_path = details.get('viz_image')
                viz_url = None
                
                if full_viz_path:
                    # Convert absolute path to relative URL
                    rel_path = os.path.relpath(full_viz_path, PROCESSED_FOLDER)
                    rel_path = rel_path.replace("\\", "/")
                    viz_url = f"processed/{rel_path}"

                size_analysis_for_web.append({
                    "filename": filename,
                    "folder": details['folder'],
                    "size_px": details['size_px'],
                    "ratio": details['ratio'],
                    "status": details['status'],
                    "visualization_url": viz_url 
                })

        # Diagnosis Summary
        overall_diagnosis = "Normal / No Parasite Detected"
        if counts['schuffner dot'] > 0: overall_diagnosis = "P. vivax Detected"
        elif counts['band form'] > 0 or counts['basket form'] > 0: overall_diagnosis = "P. malariae Detected"
        elif counts['1chromatin'] > 0: overall_diagnosis = "P. falciparum Detected"

        return jsonify({
            "session_id": session_id,
            "overall_diagnosis": overall_diagnosis,
            "total_cells_segmented": len(cropped_cell_paths),
            "vit_characteristics": analysis_results, 
            "size_analysis": size_analysis_for_web, 
            "summary": dict(counts)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath): os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5001)