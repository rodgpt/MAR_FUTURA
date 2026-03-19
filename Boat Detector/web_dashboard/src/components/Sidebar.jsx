import React from 'react';
import { Boat, MapPin, Eye, Hash, Clock } from '@phosphor-icons/react';

export default function Sidebar({
  sites,
  selectedSite,
  setSelectedSite,
  showPoints,
  setShowPoints,
  minDetections,
  setMinDetections,
  startHour,
  setStartHour,
  endHour,
  setEndHour
}) {
  return (
    <aside className="glass-panel sidebar">
      <div className="logo-container">
        <div className="logo-icon">
          <Boat weight="fill" color="white" />
        </div>
        <div className="logo-text">
          <h1>Detección de Embarcaciones</h1>
          <p>Monitoreo Acústico</p>
        </div>
      </div>

      <div className="control-group">
        <label className="control-label">
          <MapPin weight="duotone" /> Ubicación (Grupo)
        </label>
        <select 
          className="custom-select" 
          value={selectedSite} 
          onChange={(e) => setSelectedSite(e.target.value)}
        >
          {sites.map(site => (
            <option key={site} value={site}>{site}</option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label className="control-label">
          <Clock weight="duotone" /> Rango Horario (0-23)
        </label>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px' }}>
          <input 
            type="number" 
            min="0" 
            max="23"
            value={startHour}
            onChange={(e) => setStartHour(Math.min(23, Math.max(0, parseInt(e.target.value) || 0)))}
            style={{ flex: 1, borderRadius: '6px', border: '1px solid var(--panel-border)', padding: '6px 8px', background: 'var(--bg-color)', color: 'var(--text-primary)', outline: 'none' }}
          />
          <span style={{ color: 'var(--text-secondary)' }}>-</span>
          <input 
            type="number" 
            min="0" 
            max="23"
            value={endHour}
            onChange={(e) => setEndHour(Math.min(23, Math.max(0, parseInt(e.target.value) || 0)))}
            style={{ flex: 1, borderRadius: '6px', border: '1px solid var(--panel-border)', padding: '6px 8px', background: 'var(--bg-color)', color: 'var(--text-primary)', outline: 'none' }}
          />
        </div>
      </div>

      <div className="control-group">
        <label className="control-label">
          <Hash weight="duotone" /> Mínimo Detecciones/Día
        </label>
        <div className="slider-container">
          <input 
            type="range" 
            className="custom-slider" 
            min="1" 
            max="20" 
            step="1"
            value={minDetections}
            onChange={(e) => setMinDetections(Number(e.target.value))}
          />
          <span className="slider-value">{minDetections}</span>
        </div>
      </div>

      <div className="control-group" style={{ marginTop: 'auto', gap: '8px' }}>
        <label className="checkbox-group">
          <input 
            type="checkbox" 
            checked={showPoints} 
            onChange={(e) => setShowPoints(e.target.checked)} 
          />
          <div className="custom-checkbox">
            {showPoints && <Eye weight="bold" color="white" size={14} />}
          </div>
          <span style={{ fontSize: '0.9rem' }}>Mostrar puntos individuales</span>
        </label>
      </div>
    </aside>
  );
}
