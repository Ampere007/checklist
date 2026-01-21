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

# ⚠️ เช็ค Path ให้ดี
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_resnet-50_new_start.pth') 

CLASS_NAMES = ['1chromatin', 'band form', 'basket form', 'nomal_cell', 'schuffner dot'] 

# เรียกใช้ฟังก์ชันจาก model_loader.py
model, device = load_resnet_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

# --- Routes ---

@app.route('/uploads/<path:filename>')
def send_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

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
        filename = unique_id + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 1. Segmentation
        print(f"1️⃣ Running Cellpose Segmentation...")
        raw_cells_data = segment_and_save_cells(filepath)
        
        if not raw_cells_data: 
            return jsonify({'message': 'Cellpose found no cells.'})
        
        # 2. Filtering
        print(f"2️⃣ Filtering cells...")
        valid_cells_data = filter_bad_cells(raw_cells_data)
        
        if not valid_cells_data:
             return jsonify({'message': 'All cells were filtered out.'})

        # Prepare folders
        first_cell_path = valid_cells_data[0]['file_path']
        input_dir = os.path.dirname(first_cell_path)
        session_id = os.path.basename(input_dir)
        sorted_base_dir = os.path.join(PROCESSED_FOLDER, session_id, 'sorted_by_morphology')
        
        for class_name in CLASS_NAMES + ['Unknown']:
            os.makedirs(os.path.join(sorted_base_dir, class_name), exist_ok=True)

        analysis_results = []
        counts = Counter()

        # 3. Classification
        print(f"3️⃣ Classifying {len(valid_cells_data)} cells...")
        CONFIDENCE_THRESHOLD = 95.0

        for cell_item in valid_cells_data:
            cell_path = cell_item['file_path']
            bbox = cell_item['bbox']
            cell_filename = os.path.basename(cell_path)
            
            predicted_label = "Unknown"
            confidence = 0.0
            
            # Temp Mask for AI
            temp_masked_path = cell_path.replace(".png", "_temp_mask.png")
            try:
                masked_img = preprocess_image_with_mask(cell_path)
                if masked_img: masked_img.save(temp_masked_path)
                else: shutil.copy(cell_path, temp_masked_path)
            except: shutil.copy(cell_path, temp_masked_path)

            # Predict
            if model is not None:
                predicted_label, confidence = predict_image_file(model, device, temp_masked_path)
                if predicted_label != 'nomal_cell' and confidence < CONFIDENCE_THRESHOLD:
                    predicted_label = 'nomal_cell' 

            # Cleanup
            if os.path.exists(temp_masked_path):
                try: os.remove(temp_masked_path)
                except: pass

            # Copy to result folder
            target_path = os.path.join(sorted_base_dir, predicted_label, cell_filename)
            shutil.copy(cell_path, target_path)

            analysis_results.append({
                "cell": cell_filename,
                "characteristic": predicted_label,
                "confidence": f"{confidence:.2f}%", 
                "url": f"cells/{session_id}/{cell_filename}",
                "bbox": bbox
            })
            counts[predicted_label] += 1

        # 4. Size Analysis
        print(f"4️⃣ Analyzing Cell Sizes...")
        size_data_raw = process_folder_sizes(sorted_base_dir)
        size_analysis_for_web = []
        
        if size_data_raw:
            for filename, details in size_data_raw.items():
                viz_url = None
                if details.get('viz_image'):
                    rel = os.path.relpath(details['viz_image'], PROCESSED_FOLDER).replace("\\", "/")
                    viz_url = f"processed/{rel}"

                size_analysis_for_web.append({
                    "filename": filename,
                    "folder": details['folder'],
                    "size_px": details['size_px'],
                    "ratio": details['ratio'],
                    "status": details['status'],
                    "visualization_url": viz_url 
                })

        # Summary
        overall_diagnosis = "Normal / No Parasite Detected"
        if counts['schuffner dot'] > 0: overall_diagnosis = "P. vivax Detected"
        elif counts['band form'] > 0 or counts['basket form'] > 0: overall_diagnosis = "P. malariae Detected"
        elif counts['1chromatin'] > 0: overall_diagnosis = "P. falciparum Detected"

        return jsonify({
            "session_id": session_id,
            "original_image_url": f"uploads/{filename}",
            "overall_diagnosis": overall_diagnosis,
            "total_cells_segmented": len(valid_cells_data),
            "vit_characteristics": analysis_results, 
            "size_analysis": size_analysis_for_web, 
            "summary": dict(counts)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)