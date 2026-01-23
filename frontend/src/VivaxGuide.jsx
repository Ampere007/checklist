import React from 'react';
import { useNavigate } from 'react-router-dom';
import './VivaxGuide.css'; // ใช้ CSS ไฟล์ใหม่ที่ก๊อปปี้มาจากอันเดิม

// --- Import Images ---
// เปลี่ยนจาก dhaPipImage เป็น Chloroquine
import chloroquineImage from './assets/Chloroquine.png'; 
import primaquineImage from './assets/Primaquine.png';

function VivaxGuide() {
  const navigate = useNavigate();

  return (
    <div className="medication-guide-page">
      <header className="guide-header">
        {/* เปลี่ยนชื่อเชื้อเป็น Vivax */}
        <h1 className="guide-title">Treatment Guide: <i>P. vivax</i></h1>
        <p className="guide-subtitle">
          (Uncomplicated cases: Radical Cure Regimen)
        </p>
      </header>

      {/* --- Disclaimer Section (เหมือนเดิม) --- */}
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
      
      {/* --- Combo Box --- */}
      <div className="regimen-combo-box">
        {/* --- Drug 1: Chloroquine --- */}
        <div className="combo-item">
          <img src={chloroquineImage} alt="Chloroquine" className="combo-image" />
          <h4 className="combo-name">Chloroquine</h4>
          <p>(Blood Schizontocide)</p>
        </div>

        {/* --- Plus Sign --- */}
        <div className="combo-plus">
          <span>+</span>
        </div>

        {/* --- Drug 2: Primaquine --- */}
        <div className="combo-item">
          <img src={primaquineImage} alt="Primaquine" className="combo-image" />
          <h4 className="combo-name">Primaquine</h4>
          <p>(Anti-relapse / 14 Days)</p>
        </div>
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
            {/* --- Chloroquine Row (กิน 3 วันเหมือน DHA-PIP) --- */}
            <tr>
              <td>
                <strong>Chloroquine</strong>
                <br/>
                <span className="table-note">(Initial dose followed by subsequent doses)</span>
              </td>
              <td>1 Dose (Start)</td>
              <td>1 Dose (+6-8 hrs)</td>
              <td>1 Dose (Daily)</td>
            </tr>
            {/* --- Primaquine Row (กิน 14 วัน) --- */}
            <tr>
              <td>
                <strong>Primaquine</strong>
              </td>
              <td colSpan="3" className="primaquine-dose">
                <strong>1 Dose Daily for 14 Days</strong>
                <br/>
                <span className="table-note">(Start after checking G6PD status. Take with food.)</span>
              </td>
            </tr>
          </tbody>
        </table>
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

export default VivaxGuide;