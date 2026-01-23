import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FalciparumGuide.css'; // CSS remains untouched

// --- Import Images (Unchanged) ---
import dhaPipImage from './assets/eurartesim.png';
import primaquineImage from './assets/Primaquine.png';

function FalciparumGuide() {
  const navigate = useNavigate();

  return (
    <div className="medication-guide-page">
      <header className="guide-header">
        {/* Adjusted Title to English */}
        <h1 className="guide-title">Treatment Guide: <i>P. falciparum</i></h1>
        <p className="guide-subtitle">
          (Uncomplicated cases in areas without drug resistance)
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
      
      {/* --- Combo Box --- */}
      <div className="regimen-combo-box">
        {/* --- Drug 1 --- */}
        <div className="combo-item">
          <img src={dhaPipImage} alt="Dihydroartemisinin-Piperaquine" className="combo-image" />
          <h4 className="combo-name">Dihydroartemisinin-Piperaquine (DHA-PIP)</h4>
          <p>(Fixed-dose Combination)</p>
        </div>

        {/* --- Plus Sign --- */}
        <div className="combo-plus">
          <span>+</span>
        </div>

        {/* --- Drug 2 --- */}
        <div className="combo-item">
          <img src={primaquineImage} alt="Primaquine" className="combo-image" />
          <h4 className="combo-name">Primaquine</h4>
          <p>(Single Dose)</p>
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
            {/* --- DHA-PIP Row --- */}
            <tr>
              <td>
                <strong>Dihydroartemisinin-Piperaquine</strong>
                <br/>
                <span className="table-note">(Should be taken at the same time every day)</span>
              </td>
              <td>1 Dose</td>
              <td>1 Dose</td>
              <td>1 Dose</td>
            </tr>
            {/* --- Primaquine Row --- */}
            <tr>
              <td>
                <strong>Primaquine</strong>
              </td>
              <td colSpan="3" className="primaquine-dose">
                <strong>Single dose on any one day</strong>
                <br/>
                <span className="table-note">(Provided the patient can tolerate oral intake without vomiting)</span>
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

export default FalciparumGuide;