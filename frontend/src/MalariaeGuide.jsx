import React from 'react';
import { useNavigate } from 'react-router-dom';
import './MalariaeGuide.css'; // ใช้ CSS ไฟล์ใหม่

// --- Import Images ---
// Malariae ใช้แค่ Chloroquine ตัวเดียว
import chloroquineImage from './assets/Chloroquine.png'; 

function MalariaeGuide() {
  const navigate = useNavigate();

  return (
    <div className="medication-guide-page">
      <header className="guide-header">
        {/* เปลี่ยนชื่อเชื้อเป็น Malariae */}
        <h1 className="guide-title">Treatment Guide: <i>P. malariae</i></h1>
        <p className="guide-subtitle">
          (Uncomplicated cases: Blood Schizontocide Only)
        </p>
      </header>

      {/* --- Disclaimer Section --- */}
      <div className="disclaimer-section">
        <div className="disclaimer-card">
          <h4 className="disclaimer-title">⚠️ Important Medical Disclaimer</h4>
          <p className="disclaimer-text">
            This information is a summary example and **does NOT constitute medical advice.**
            Malaria treatment is highly complex and must be under the strict supervision of a doctor.
            **Do NOT purchase medication for self-treatment.**
          </p>
        </div>
      </div>

      {/* --- Recommended Regimen Section --- */}
      <h2 className="regimen-title">Recommended Regimen</h2>
      
      {/* --- Single Drug Box (No Combo) --- */}
      <div className="regimen-combo-box single-drug-mode">
        {/* --- Drug 1: Chloroquine --- */}
        <div className="combo-item">
          <img src={chloroquineImage} alt="Chloroquine" className="combo-image" />
          <h4 className="combo-name">Chloroquine</h4>
          <p>(Blood Schizontocide / 3 Days)</p>
        </div>
        
        {/* ลบเครื่องหมาย + และ Primaquine ออก เพราะ Malariae ไม่ต้องใช้ยาฆ่าเชื้อในตับ */}
      </div>

      {/* --- Administration Table --- */}
      <h2 className="regimen-title">Administration</h2>
      
      <div className="table-wrapper">
        <table className="medication-table">
          <thead>
            <tr>
              {/* Table Headers */}
              <th>Day relative to treatment start</th>
              <th>Day 0 (Start)</th>
              <th>Day 1</th>
              <th>Day 2</th>
            </tr>
          </thead>
          <tbody>
            {/* --- Chloroquine Row Only --- */}
            <tr>
              <td>
                <strong>Chloroquine</strong>
                <br/>
                <span className="table-note">(Total course: 3 Days)</span>
              </td>
              <td>
                <strong>Initial Dose</strong><br/>
                <small>followed by 2nd dose (+6-8 hrs)</small>
              </td>
              <td>1 Dose (Daily)</td>
              <td>1 Dose (Daily)</td>
            </tr>
            {/* ตัดแถว Primaquine ออก */}
          </tbody>
        </table>
      </div>

      {/* --- Note Section (Specific to Malariae) --- */}
      <div className="note-section">
        <p>
          <strong>Note:</strong> <i>P. malariae</i> does not have a dormant liver stage (hypnozoites). 
          Therefore, <strong>Primaquine is NOT required</strong> for radical cure, unlike in <i>P. vivax</i> or <i>P. ovale</i> infections.
        </p>
      </div>

      {/* --- Back Button --- */}
      <div className="back-button-container">
        <button onClick={() => navigate(-1)} className="back-button">
          &larr; Back to Analysis Results
        </button>
      </div>
    </div>
  );
}

export default MalariaeGuide;