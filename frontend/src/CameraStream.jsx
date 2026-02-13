import React, { useState, useEffect } from 'react';
import { initializeApp, getApps, getApp } from "firebase/app";
import { getDatabase, ref, onValue } from "firebase/database";

// --- ส่วน Config ของคุณ (ใส่ให้ครบแล้ว) ---
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

// --- เริ่มต้น Firebase (แบบป้องกันการ Init ซ้ำ) ---
let app;
if (!getApps().length) {
    app = initializeApp(firebaseConfig);
} else {
    app = getApp(); // ถ้ามีอยู่แล้วให้ใช้ตัวเดิม
}
const db = getDatabase(app);

const CameraStream = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    // อ้างอิงไปที่ streams/stream1 (ที่ Raspberry Pi ส่งมา)
    const streamRef = ref(db, 'streams/stream1');

    // ดักฟังข้อมูลเมื่อมีการเปลี่ยนแปลง
    const unsubscribe = onValue(streamRef, (snapshot) => {
      const data = snapshot.val();
      if (data && data.frame) {
        setImageSrc(data.frame); // อัปเดตภาพ Base64
        setLastUpdate(new Date(data.ts).toLocaleTimeString()); // อัปเดตเวลา
      }
    });

    return () => unsubscribe(); // ล้างการทำงานเมื่อปิด Component
  }, []);

  return (
    <div style={{ textAlign: 'center', marginTop: '20px' }}>
      <h3>🔴 กล้อง Raspberry Pi (Live)</h3>
      <div style={{ 
        border: '5px solid #333', 
        borderRadius: '10px',
        display: 'inline-block',
        overflow: 'hidden',
        boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
        background: '#000'
      }}>
        {imageSrc ? (
          <img src={imageSrc} alt="Live Stream" style={{ width: '640px', maxWidth: '100%', display: 'block' }} />
        ) : (
          <div style={{ width: '640px', height: '480px', color: '#fff', display: 'flex', flexDirection:'column', alignItems: 'center', justifyContent: 'center' }}>
            <div className="loader" style={{marginBottom:'10px'}}></div>
            กำลังรอสัญญาณภาพ...
          </div>
        )}
      </div>
      <p style={{ color: '#666', fontSize: '0.9rem' }}>อัปเดตล่าสุด: {lastUpdate || "รอการเชื่อมต่อ"}</p>
    </div>
  );
};

export default CameraStream;