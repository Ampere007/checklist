import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import { Routes, Route, Link } from 'react-router-dom';
import { initializeApp } from "firebase/app";
import { getDatabase, ref, onValue, push, set } from "firebase/database";

// --- IMPORT GUIDES ---
import FalciparumGuide from './FalciparumGuide'; 
import VivaxGuide from './VivaxGuide';
import MalariaeGuide from './MalariaeGuide'; 

// --- IMPORT ICONS ---
import checkIcon from './picture/check (3).png';
import emptyBoxIcon from './picture/check-box-empty.png';

const BACKEND_URL = 'http://127.0.0.1:5001';

// ==============================
// 0. FIREBASE CONFIGURATION
// ==============================
const firebaseConfig = {
  apiKey: "AIzaSyDa_8UHDLV8i4h7jdsm-fEHNMgW-h61p04",
  authDomain: "malariaxchecklist.firebaseapp.com",
  databaseURL: "https://malariaxchecklist-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "malariaxchecklist",
  storageBucket: "malariaxchecklist.firebasestorage.app",
  messagingSenderId: "528337272211",
  appId: "1:528337272211:web:b030370ae52bff5d7afc66",
  measurementId: "G-YL05EPR7PT"
};

let app, db;
try {
    app = initializeApp(firebaseConfig);
    db = getDatabase(app);
} catch (err) {
    console.log("Firebase already initialized, using existing app.");
    app = initializeApp(firebaseConfig, "secondary");
    db = getDatabase(app);
}

// ==============================
// 1. HELPER FUNCTIONS
// ==============================
const dataURLtoFile = (dataurl, filename) => {
    let arr = dataurl.split(','),
        mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), 
        n = bstr.length, 
        u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
}

// ==============================
// 2. COMPONENTS
// ==============================

const InteractiveImage = ({ imageUrl, cells, onCellClick }) => {
    const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
    const imgRef = useRef(null);

    const handleImageLoad = (e) => {
        setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight });
    };

    return (
        <div style={{ position: 'relative', width: '100%', overflow: 'hidden', borderRadius: '16px' }}>
            <img 
                ref={imgRef} src={imageUrl} alt="Analyzed Slide" 
                onLoad={handleImageLoad} style={{ width: '100%', display: 'block' }} 
            />
            {imgSize.w > 0 && cells.map((cell, index) => {
                if (!cell.bbox) return null;
                // คำนวณตำแหน่ง % เทียบกับขนาดรูปจริงที่โหลดมา (ซึ่งตอนนี้จะเป็นรูปสี่เหลี่ยมแล้ว)
                const left = (cell.bbox.x / imgSize.w) * 100;
                const top = (cell.bbox.y / imgSize.h) * 100;
                const width = (cell.bbox.w / imgSize.w) * 100;
                const height = (cell.bbox.h / imgSize.h) * 100;

                return (
                    <div key={index} onClick={() => onCellClick(cell)} className="interactive-box"
                        style={{ position: 'absolute', left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}>
                        <div className="box-tooltip">{cell.characteristic}</div>
                    </div>
                );
            })}
        </div>
    );
};

// Galleries
const ImageGallery = ({ title, images, onClose }) => (
    <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
                <h3>{title} <span style={{color:'#64748b', fontSize:'0.9em'}}>({images.length})</span></h3>
                <button onClick={onClose} className="icon-close-btn">×</button>
            </div>
            <div className="abnormal-cells-grid">
                {images.map((item, index) => (
                    <div key={index} className="abnormal-cell-item">
                        <img src={`${BACKEND_URL}/${item.url}`} alt={`Cell ${index + 1}`} />
                        <p>{item.characteristic}</p>
                    </div>
                ))}
            </div>
            <div style={{textAlign:'center', marginTop:'30px'}}>
                <button onClick={onClose} className="close-button">ปิดหน้าต่าง</button>
            </div>
        </div>
    </div>
);

const SizeGallery = ({ title, items, onClose }) => (
    <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
                <h3>{title}</h3>
                <p style={{color:'#64748b', fontSize:'0.9rem', marginTop:'5px'}}>ภาพแสดงเส้นขอบ (สีเขียว) และวงกลมคำนวณ (สีเหลือง)</p>
            </div>
            <div className="abnormal-cells-grid">
                {items.map((item, index) => (
                    <div key={index} className="abnormal-cell-item" style={{ border: item.status === 'Enlarged' ? '2px solid #ef4444' : '1px solid #e2e8f0' }}>
                        <div style={{position:'relative', overflow: 'hidden', borderRadius: '8px 8px 0 0'}}>
                            <img src={`${BACKEND_URL}/${item.visualization_url}`} alt="Size Viz" style={{width: '100%', display: 'block'}} />
                            {item.status === 'Enlarged' && <div className="enlarged-label-overlay">{item.folder}</div>}
                            {item.status === 'Enlarged' && <span className="badge-enlarged">Enlarged</span>}
                        </div>
                        <div style={{padding:'12px', textAlign:'left', background:'white'}}>
                            <div className="info-row"><span className="label">Size:</span> <strong>{item.size_px} px</strong></div>
                            <div className="info-row"><span className="label">Ratio:</span> <strong style={{color: item.ratio > 1.2 ? '#ef4444' : '#10b981'}}>{item.ratio}x</strong></div>
                            <div className="info-row border-top"><span className="label">Shape:</span> <strong style={{color: item.shape === 'Amoeboid' ? '#8b5cf6' : '#64748b'}}>{item.shape || 'Round'}</strong></div>
                        </div>
                    </div>
                ))}
            </div>
            <div style={{textAlign:'center', marginTop:'30px'}}>
                <button onClick={onClose} className="close-button">ปิดหน้าต่าง</button>
            </div>
        </div>
    </div>
);

const DistanceGallery = ({ title, items, onClose }) => (
    <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
                <h3>{title}</h3>
                <p style={{color:'#64748b', fontSize:'0.9rem', marginTop:'5px'}}>ภาพจำลองอัลกอริทึมวัดระยะห่าง (พื้นหลังดำ)</p>
            </div>
            <div className="abnormal-cells-grid">
                {items.map((item, index) => (
                    <div key={index} className="abnormal-cell-item" style={{ border: '2px solid #0ea5e9' }}>
                        <div style={{position:'relative', overflow: 'hidden', borderRadius: '8px 8px 0 0', background:'black'}}>
                            <img src={`${BACKEND_URL}/${item.distance_viz_url}`} alt="Algo Viz" style={{width: '100%', display: 'block'}} />
                        </div>
                        <div style={{padding:'12px', textAlign:'left', background:'white'}}>
                            <div className="info-row"><span className="label">Type:</span> <strong style={{color:'var(--primary)'}}>{item.characteristic}</strong></div>
                            <div className="info-row"><span className="label">Marginal Ratio:</span> <strong style={{color: item.marginal_ratio > 0.75 ? '#ef4444' : '#0369a1'}}>{(item.marginal_ratio * 100).toFixed(1)}%</strong></div>
                            <div className="info-row border-top"><span className="label">Pos:</span> <span style={{fontSize:'0.8rem', color:'#64748b'}}>{item.marginal_ratio > 0.75 ? "Edge (Appliqué)" : "Internal"}</span></div>
                        </div>
                    </div>
                ))}
            </div>
            <div style={{textAlign:'center', marginTop:'30px'}}>
                <button onClick={onClose} className="close-button">ปิดหน้าต่าง</button>
            </div>
        </div>
    </div>
);

const SingleImageViewer = ({ imageUrl, onClose }) => {
    if (!imageUrl) return null;
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="single-image-modal" onClick={(e) => e.stopPropagation()}>
          <img src={imageUrl} alt="Result" />
          <button onClick={onClose} className="close-button" style={{marginTop:'20px'}}>ปิดรูปภาพ</button>
        </div>
      </div>
    );
};

// ==============================
// 3. MAIN PAGE LOGIC
// ==============================

function AnalysisPage() {
  const [inputMode, setInputMode] = useState('choose'); 
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [res, setRes] = useState(null); 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Camera State
  const [liveImage, setLiveImage] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Galleries States
  const [showChromatinGallery, setShowChromatinGallery] = useState(false);
  const [showSchuffnerGallery, setShowSchuffnerGallery] = useState(false);
  const [showBasketGallery, setShowBasketGallery] = useState(false);
  const [showSizeGallery, setShowSizeGallery] = useState(false);
  const [showDistanceGallery, setShowDistanceGallery] = useState(false);
  
  const [viewingImage, setViewingImage] = useState(null);
  const [selectedCellDetail, setSelectedCellDetail] = useState(null);
  const [modalImgSize, setModalImgSize] = useState({ w: 0, h: 0 });

  // --- CAMERA LOGIC ---
  useEffect(() => {
    if (inputMode === 'camera' && !res) {
        const streamRef = ref(db, 'streams/stream1');
        const unsubscribe = onValue(streamRef, (snapshot) => {
            const data = snapshot.val();
            if (data && data.frame) {
                setLiveImage(data.frame);
                setLastUpdate(new Date(data.ts).toLocaleTimeString());
            }
        });
        return () => unsubscribe();
    }
  }, [inputMode, res]);

  // --- HANDLERS ---
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setRes(null); setError('');
      if (preview) URL.revokeObjectURL(preview);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleCaptureAndAnalyze = async () => {
      if (!liveImage) return setError("ยังไม่ได้รับสัญญาณภาพจากกล้อง");
      
      setLoading(true); setError('');
      
      const file = dataURLtoFile(liveImage, "capture_rpi.jpg");
      
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));

      try {
          const capturesRef = ref(db, 'captures');
          const newCaptureRef = push(capturesRef);
          await set(newCaptureRef, {
              image: liveImage,
              timestamp: Date.now(),
              date: new Date().toLocaleString(),
              note: "Captured via Web Interface"
          });
      } catch (dbError) {
          console.error("บันทึกลง Firebase ไม่สำเร็จ:", dbError);
      }
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await axios.post(`${BACKEND_URL}/api/analyze`, formData);
        setRes(response.data);
      } catch (err) {
        console.error(err);
        setError("เกิดข้อผิดพลาด: Backend ไม่ตอบสนอง");
      } finally {
        setLoading(false);
      }
  };

  const handleSubmit = async () => { 
    if (!selectedFile) return setError('กรุณาเลือกไฟล์ก่อน');
    setLoading(true); setError(''); setRes(null); 
    const formData = new FormData(); 
    formData.append('file', selectedFile);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/analyze`, formData);
      setRes(response.data);
    } catch (err) {
      console.error(err);
      setError("เกิดข้อผิดพลาด: Backend ไม่ตอบสนอง"); 
    } finally {
      setLoading(false);
    }
  };

  // --- DATA PROCESSING ---
  const overallDiagnosis = res?.overall_diagnosis || "";
  const totalCells = res?.total_cells_segmented || 0;
  const allCells = res?.vit_characteristics || [];
  
  const sizeData = (res?.size_analysis || []).filter(item => {
      const url = (item.visualization_url || "").toLowerCase();
      return !url.includes('dist_viz') && !url.includes('distance');
  });

  const amoeboidCount = res?.amoeboid_count || 0;
  const isVivax = overallDiagnosis.includes("vivax");
  const isFalciparum = overallDiagnosis.includes("falciparum");
  const isMalariae = overallDiagnosis.includes("malariae");

  const chromatinCells = allCells.filter(c => c.characteristic === '1chromatin');
  const schuffnerCells = allCells.filter(c => ['schuffner dot'].includes(c.characteristic)); 
  const basketCells = allCells.filter(c => ['band form', 'basket form'].includes(c.characteristic));
  const abnormalCells = allCells.filter(c => c.characteristic !== 'nomal_cell');
  const distanceData = allCells.filter(c => c.distance_viz_url);

  let avgRatio = 1.0;
  if (sizeData.length > 0) {
      const sum = sizeData.reduce((acc, curr) => acc + curr.ratio, 0);
      avgRatio = (sum / sizeData.length).toFixed(2);
  }

  // Checklist Flags
  const chkChromatin = isFalciparum; 
  const chkApplique = isFalciparum; 
  const chkSchuffner = isVivax;
  const chkAmoeboid = amoeboidCount > 0 || isVivax;
  const chkEnlarged = sizeData.some(item => item.status === 'Enlarged'); 
  const chkSmaller = isMalariae;
  const chkBand = allCells.some(c => c.characteristic === 'band form');
  const chkBasket = allCells.some(c => c.characteristic === 'basket form');

  // =========================================================
  // 🔥 FIX: Determine which image to show in Main Result 🔥
  // =========================================================
  // ถ้า Backend ส่ง original_image_url มา (เป็นรูปที่ Crop แล้ว) ให้ใช้รูปนั้น
  // ถ้าไม่มี (ยังไม่ได้กดวิเคราะห์) ให้ใช้ preview (รูปต้นฉบับในเครื่อง)
  const finalDisplayImage = res?.original_image_url 
      ? `${BACKEND_URL}/${res.original_image_url}` 
      : preview;

  // --- RENDER ---
  return (
    <div className="container"> 
       <header className="header">
        <h1>MALA-Sight</h1>
        <p>AI-Powered Malaria Screening & Morphometric Analysis</p>
      </header>

      {/* INPUT SECTION */}
      {!res && !loading && (
        <div className="upload-section">
            {/* 1. หน้าเลือกโหมด */}
            {inputMode === 'choose' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', maxWidth: '600px', margin: '0 auto' }}>
                    <button onClick={() => setInputMode('camera')} className="mode-btn" style={{ padding: '40px', fontSize: '1.2rem', cursor: 'pointer', borderRadius: '12px', border: '2px solid #333', background: '#fff' }}>
                        <div style={{fontSize: '3rem', marginBottom: '10px'}}>📷</div><b>ใช้กล้อง RPi (Live)</b>
                    </button>
                    <button onClick={() => setInputMode('upload')} className="mode-btn" style={{ padding: '40px', fontSize: '1.2rem', cursor: 'pointer', borderRadius: '12px', border: '2px solid #333', background: '#fff' }}>
                        <div style={{fontSize: '3rem', marginBottom: '10px'}}>📁</div><b>อัปโหลดไฟล์</b>
                    </button>
                </div>
            )}

            {/* 2. โหมดอัปโหลด */}
            {inputMode === 'upload' && (
                <>
                    <input type="file" onChange={handleFileChange} accept="image/*" className="file-input" id="file"/>
                    {preview ? (
                        <div>
                            <img src={preview} alt="Preview" className="upload-preview"/>
                            <div><label htmlFor="file" className="file-label">เปลี่ยนรูปภาพ</label></div>
                        </div>
                    ) : (
                        <label htmlFor="file" style={{cursor:'pointer'}}>
                            <div className="upload-icon">+</div><h3>อัปโหลดภาพฟิล์มเลือด</h3><div className="file-label">เลือกไฟล์รูปภาพ</div>
                        </label>
                    )}
                    <div style={{marginTop: '30px', display:'flex', gap:'10px', justifyContent:'center'}}>
                        <button onClick={() => {setInputMode('choose'); setPreview(null); setSelectedFile(null);}} className="restart-btn" style={{background:'#64748b', color:'white'}}>← ย้อนกลับ</button>
                        {selectedFile && (<button onClick={handleSubmit} className="analyze-button">เริ่มวิเคราะห์ภาพ</button>)}
                    </div>
                </>
            )}

            {/* 3. โหมดกล้อง */}
            {inputMode === 'camera' && (
                <div style={{textAlign: 'center'}}>
                    <div style={{ border: '5px solid #333', borderRadius: '10px', display: 'inline-block', overflow: 'hidden', background: '#000', boxShadow: '0 4px 8px rgba(0,0,0,0.2)', marginBottom: '20px' }}>
                        {liveImage ? (
                            <img src={liveImage} alt="Live Stream" style={{ width: '450x', height: '420px', objectFit: 'cover', transform: 'scale(1.3)', display: 'block' }} />
                        ) : (
                            <div style={{ width: '640px', height: '480px', color: '#fff', display: 'flex', flexDirection:'column', alignItems: 'center', justifyContent: 'center' }}>
                                <div className="loader" style={{width:'30px', height:'30px', marginBottom:'10px'}}></div>กำลังเชื่อมต่อกล้อง Raspberry Pi...
                            </div>
                        )}
                    </div>
                    {lastUpdate && <p style={{color:'#64748b', fontSize:'0.8rem', marginTop:'-15px', marginBottom:'20px'}}>อัปเดตล่าสุด: {lastUpdate}</p>}
                    <div style={{display:'flex', gap:'15px', justifyContent:'center'}}>
                         <button onClick={() => {setInputMode('choose'); setLiveImage(null);}} className="restart-btn" style={{background:'#64748b', color:'white'}}>← ยกเลิก</button>
                        <button onClick={handleCaptureAndAnalyze} disabled={!liveImage} className="analyze-button" style={{ background: liveImage ? '#ef4444' : '#ccc', cursor: liveImage ? 'pointer' : 'not-allowed' }}>📸 ถ่ายภาพ & วิเคราะห์ทันที</button>
                    </div>
                </div>
            )}
            {error && <p className="error-msg">{error}</p>}
        </div>
      )}

      {loading && (
          <div className="loader-container">
              <div className="loader"></div>
              <h3>AI กำลังประมวลผล...</h3>
              <p>Cell Segmentation &rarr; Feature Extraction &rarr; Diagnosis</p>
          </div>
      )}

      {res && (
        <div className="results-grid">
          {/* Left Column */}
          <div className="results-left">
             <div className="detail-card">
                <h4>ภาพที่วิเคราะห์ <span style={{fontSize:'0.8rem', color:'#64748b'}}>(แตะกรอบแดงเพื่อดูรายละเอียด)</span></h4>
                
                {/* 🔥 FIX: ใช้รูป finalDisplayImage แทน preview 
                   เพื่อให้รูปสี่เหลี่ยมที่ตัดแล้ว แสดงตรงกับพิกัด Bounding Box
                */}
                {finalDisplayImage && (
                    <InteractiveImage 
                        imageUrl={finalDisplayImage} 
                        cells={abnormalCells} 
                        onCellClick={(cell) => setSelectedCellDetail(cell)} 
                    />
                )}

             </div>

             <div className="detail-card clickable-card" onClick={() => setShowSizeGallery(true)} style={{borderLeft:'5px solid #8b5cf6'}}>
                 <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                    <div><h4 style={{marginBottom:5, border:0, color:'#8b5cf6'}}>Size Analysis Visualization</h4><p style={{margin:0, fontSize:'0.85rem', color:'#64748b'}}>ตรวจพบ {sizeData.length} เซลล์ (Size)</p></div>
                    <span className="arrow-icon">→</span>
                 </div>
                 <div className="mini-gallery">
                    {sizeData.slice(0, 4).map((item, idx) => (
                        <img key={idx} src={`${BACKEND_URL}/${item.visualization_url}`} alt="viz" />
                    ))}
                 </div>
             </div>

             {distanceData.length > 0 && (
                 <div className="detail-card clickable-card" onClick={() => setShowDistanceGallery(true)} style={{borderLeft:'5px solid #0ea5e9'}}>
                     <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                        <div><h4 style={{marginBottom:5, border:0, color:'#0ea5e9'}}>Distance Algo Visualization</h4><p style={{margin:0, fontSize:'0.85rem', color:'#64748b'}}>ตรวจพบ {distanceData.length} เชื้อ (Algorithm Line)</p></div>
                        <span className="arrow-icon">→</span>
                     </div>
                     <div className="mini-gallery">
                        {distanceData.slice(0, 4).map((item, idx) => (
                            <img key={idx} src={`${BACKEND_URL}/${item.distance_viz_url}`} alt="dist-viz" style={{border:'1px solid #bae6fd'}} />
                        ))}
                     </div>
                 </div>
             )}

             {chromatinCells.length > 0 && (<div className="detail-card clickable-card" onClick={() => setShowChromatinGallery(true)}><h4>Multi-Chromatin <span style={{color:'#64748b'}}>({chromatinCells.length})</span></h4></div>)}
             {schuffnerCells.length > 0 && (<div className="detail-card clickable-card" onClick={() => setShowSchuffnerGallery(true)}><h4>Schüffner's Dot <span style={{color:'#64748b'}}>({schuffnerCells.length})</span></h4></div>)}
             {basketCells.length > 0 && (<div className="detail-card clickable-card" onClick={() => setShowBasketGallery(true)}><h4>Basket / Band Form <span style={{color:'#64748b'}}>({basketCells.length})</span></h4></div>)}
          </div>

          {/* Right Column */}
          <div className="results-right">
             <div className="overall-diagnosis-card">
                <h4>DIAGNOSIS RESULT</h4>
                <h2 style={{color: isFalciparum?'#fca5a5': (isVivax?'#86efac': (isMalariae ? '#fdba74' : '#ffffff'))}}>{overallDiagnosis}</h2>
                <div className="checklist-grid-container">
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '15px' }}>
                      <div className="checklist-column" style={{opacity: isFalciparum ? 1 : 0.4}}>
                        <h5 style={{ color: '#fca5a5' }}>P. falciparum</h5>
                        <div className="checklist-item"><img src={chkChromatin ? checkIcon : emptyBoxIcon} width="22" /> Chromatin</div>
                        <div className="checklist-item"><img src={chkApplique ? checkIcon : emptyBoxIcon} width="22" /> Appliqué</div>
                      </div>
                      <div className="checklist-column" style={{opacity: isVivax ? 1 : 0.4}}>
                        <h5 style={{color: '#86efac'}}>P. vivax</h5>
                        <div className="checklist-item"><img src={chkSchuffner ? checkIcon : emptyBoxIcon} width="22" /> Schüffner</div>
                        <div className="checklist-item"><img src={chkEnlarged ? checkIcon : emptyBoxIcon} width="22" /> Enlarged</div>
                        <div className="checklist-item"><img src={chkAmoeboid ? checkIcon : emptyBoxIcon} width="22" /> Amoeboid</div>
                      </div>
                      <div className="checklist-column" style={{opacity: isMalariae ? 1 : 0.4}}>
                        <h5 style={{color: '#fdba74'}}>P. malariae</h5>
                        <div className="checklist-item"><img src={chkSmaller ? checkIcon : emptyBoxIcon} width="22" /> Smaller</div>
                        <div className="checklist-item"><img src={chkBand ? checkIcon : emptyBoxIcon} width="22" /> Band Form</div>
                        <div className="checklist-item"><img src={chkBasket ? checkIcon : emptyBoxIcon} width="22" /> Basket Form</div>
                      </div>
                    </div>
                </div>
             </div>

             <div className="stats-grid">
                <div className='detail-card center-text'><p>Total Cells</p><strong>{totalCells}</strong></div>
                <div className='detail-card center-text'><p>Abnormal Found</p><strong style={{color:'#f59e0b'}}>{abnormalCells.length}</strong></div>
             </div>

             <div className='detail-card' style={{borderLeft:'5px solid var(--primary)'}}>
                  <h4>Morphometry Analysis</h4>
                  <div style={{marginTop:'15px'}}>
                      <div style={{display:'flex', justifyContent:'space-between', marginBottom:'10px'}}><span>Avg Size Ratio (Cell / RBC)</span><strong style={{color: parseFloat(avgRatio) > 1.2 ? '#ef4444' : '#10b981'}}>{avgRatio}x</strong></div>
                      <div className="morph-bar-bg"><div className="morph-bar-fill" style={{ width: `${Math.min(parseFloat(avgRatio) / 2.5 * 100, 100)}%` }}></div></div>
                      <div className="morph-labels"><span>Normal (1.0x)</span><span>Enlarged (1.5x)</span></div>
                  </div>
             </div>

             <div className="action-buttons">
                <button onClick={()=>window.location.reload()} className="restart-btn">เริ่มใหม่</button>
                {isFalciparum && <Link to="/medication-guide/falciparum"><button className="med-btn-red">แนวทางการรักษา (P.f)</button></Link>}
                {isVivax && <Link to="/medication-guide/vivax"><button className="med-btn-green">แนวทางการรักษา (P.v)</button></Link>}
                {isMalariae && <Link to="/medication-guide/malariae"><button className="med-btn-orange">แนวทางการรักษา (P.m)</button></Link>}
             </div>
          </div>
        </div>
      )}

      {/* GALLERIES & MODALS */}
      {showSizeGallery && <SizeGallery title="Size Calculation Results" items={sizeData} onClose={()=>setShowSizeGallery(false)} />}
      {showDistanceGallery && <DistanceGallery title="Distance Algorithm Visualization" items={distanceData} onClose={()=>setShowDistanceGallery(false)} />}
      {showChromatinGallery && <ImageGallery title="Abnormal Chromatin" images={chromatinCells} onClose={()=>setShowChromatinGallery(false)}/>}
      {showSchuffnerGallery && <ImageGallery title="Schüffner's Dot / Amoeboid" images={schuffnerCells} onClose={()=>setShowSchuffnerGallery(false)}/>}
      {showBasketGallery && <ImageGallery title="Basket/Band Form" images={basketCells} onClose={()=>setShowBasketGallery(false)}/>}
      {viewingImage && <SingleImageViewer imageUrl={viewingImage} onClose={()=>setViewingImage(null)}/>}

      {/* CELL DETAIL MODAL */}
      {selectedCellDetail && (
          <div className="modal-overlay" onClick={() => setSelectedCellDetail(null)}>
              <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{textAlign:'center', maxWidth:'900px', padding: '30px'}}>
                  <h3 style={{color:'#ef4444', marginTop:0, marginBottom: '20px'}}>ตรวจพบเชื้อผิดปกติ!</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px', marginBottom: '25px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <p style={{fontSize:'0.8rem', color:'#64748b', marginBottom:'5px', fontWeight:'600'}}>1. Original Detection</p>
                          <div style={{ position: 'relative', width: '100%', aspectRatio: '1', borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0' }}>
                              <img src={`${BACKEND_URL}/${selectedCellDetail.url}`} alt="Original" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onLoad={(e) => setModalImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
                              {selectedCellDetail.chromatin_bboxes && selectedCellDetail.chromatin_bboxes.map((box, idx) => {
                                  const [x1, y1, x2, y2] = box;
                                  const top = (y1 / modalImgSize.h) * 100;
                                  const left = (x1 / modalImgSize.w) * 100;
                                  const width = ((x2 - x1) / modalImgSize.w) * 100;
                                  const height = ((y2 - y1) / modalImgSize.h) * 100;
                                  return (<div key={idx} style={{ position: 'absolute', top: `${top}%`, left: `${left}%`, width: `${width}%`, height: `${height}%`, border: '2px solid #00ff00', borderRadius: '2px', boxShadow: '0 0 3px rgba(0,0,0,0.5)', pointerEvents: 'none' }}></div>);
                              })}
                          </div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <p style={{fontSize:'0.8rem', color:'#64748b', marginBottom:'5px', fontWeight:'600'}}>2. Size Analysis</p>
                          <div style={{ width: '100%', aspectRatio: '1', borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0', background: '#f8fafc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              {(() => {
                                  const matched = sizeData.find(item => item.filename === selectedCellDetail.cell);
                                  return matched && matched.visualization_url ? (<img src={`${BACKEND_URL}/${matched.visualization_url}`} alt="Size Viz" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />) : <span style={{color:'#94a3b8', fontSize:'0.8rem'}}>No Size Data</span>;
                              })()}
                          </div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <p style={{fontSize:'0.8rem', color:'#64748b', marginBottom:'5px', fontWeight:'600'}}>3. Distance Algorithm</p>
                          <div style={{ width: '100%', aspectRatio: '1', borderRadius: '12px', overflow: 'hidden', border: '2px solid #bae6fd', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              {selectedCellDetail.distance_viz_url ? (<img src={`${BACKEND_URL}/${selectedCellDetail.distance_viz_url}`} alt="Algo Viz" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />) : (<span style={{color:'#475569', fontSize:'0.8rem'}}>N/A</span>)}
                          </div>
                      </div>
                  </div>
                  <div className="modal-info-container">
                      <p style={{margin:'0 0 15px 0', fontSize:'1.15rem', textAlign: 'center'}}><strong>ประเภทเชื้อ:</strong> <span style={{color:'var(--primary)', fontWeight: '700'}}>{selectedCellDetail.characteristic}</span></p>
                      <div className="modal-analysis-grid-four">
                          <div className="analysis-box"><span className="analysis-label">Morphometry</span>{(() => {const matched = sizeData.find(item => item.filename === selectedCellDetail.cell); return (<div className="analysis-value-wrapper"><strong className="analysis-main-value">{matched ? `${matched.size_px} px` : "N/A"}</strong><span className="analysis-sub-value" style={{ color: matched?.ratio > 1.2 ? '#ef4444' : '#10b981' }}>({matched ? `${matched.ratio}x` : "1.0x"})</span></div>);})()}</div>
                          <div className="analysis-box"><span className="analysis-label">Morphology / Count</span>{(() => {const matched = sizeData.find(item => item.filename === selectedCellDetail.cell); const shape = matched?.shape || "Round/Normal"; const count = selectedCellDetail.chromatin_count || 0; const isChromatin = selectedCellDetail.characteristic === '1chromatin'; return (<div style={{display:'flex', flexDirection:'column', alignItems:'center'}}><strong className="analysis-main-value" style={{ color: shape === 'Amoeboid' ? '#8b5cf6' : '#475569' }}>{shape}</strong>{isChromatin && (<span style={{fontSize:'0.75rem', color: count > 1 ? '#ef4444' : '#10b981', fontWeight:'600'}}>{count > 1 ? `Multiple (${count})` : `Single (${count})`}</span>)}</div>);})()}</div>
                          <div className="analysis-box" style={{background: '#f0f9ff', border: '1px solid #bae6fd'}}><span className="analysis-label" style={{color: '#0369a1'}}>Distance Pos</span>{selectedCellDetail.characteristic === '1chromatin' ? (<div style={{textAlign:'center'}}><strong className="analysis-main-value" style={{color:'#0c4a6e'}}>{(selectedCellDetail.marginal_ratio * 100).toFixed(1)}%</strong><p style={{fontSize:'0.65rem', margin:0, color:'#0369a1'}}>{selectedCellDetail.marginal_ratio > 0.75 ? "Edge (Appliqué)" : "Internal"}</p></div>) : <span style={{color:'#94a3b8', fontSize:'0.9rem'}}>N/A</span>}</div>
                          <div className="analysis-box" style={{background: '#fef2f2', border: '1px solid #fee2e2'}}><span className="analysis-label" style={{color: '#b91c1c'}}>Status</span>{(() => {const matched = sizeData.find(item => item.filename === selectedCellDetail.cell); const isLarge = matched?.ratio > 1.2; const isAmoeboid = matched?.shape === 'Amoeboid'; const count = selectedCellDetail.chromatin_count || 0; const isMulti = count > 1; let statusText = "Abnormal"; if (isLarge && isAmoeboid) statusText = "High Risk (P.v)"; else if (isMulti) statusText = "Multiple Infection"; else if (isLarge) statusText = "Enlarged RBC"; else if (isAmoeboid) statusText = "Amoeboid Form"; return <strong className="analysis-main-value" style={{ color: '#ef4444', fontSize: '0.9rem' }}>{statusText}</strong>;})()}</div>
                      </div>
                      {selectedCellDetail.bbox && (<div className="modal-footer-info">ตำแหน่งพิกัด: X={selectedCellDetail.bbox.x.toFixed(0)}, Y={selectedCellDetail.bbox.y.toFixed(0)}</div>)}
                  </div>
                  <button onClick={() => setSelectedCellDetail(null)} className="close-button-alt">ปิดหน้าต่าง</button>
              </div>
          </div>
      )}
    </div> 
  );
}

function AppWrapper() { 
  return (
    <div className="app-container">
      <Routes>
        <Route path="/" element={<AnalysisPage />} /> 
        <Route path="/medication-guide/falciparum" element={<FalciparumGuide />} /> 
        <Route path="/medication-guide/vivax" element={<VivaxGuide />} />
        <Route path="/medication-guide/malariae" element={<MalariaeGuide />} />
      </Routes>
    </div>
  );
}

export default AppWrapper;