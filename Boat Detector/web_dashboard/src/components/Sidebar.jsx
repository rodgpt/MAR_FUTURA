import React from 'react';
import { Boat, MapPin, Calendar, SlidersHorizontal, Eye } from '@phosphor-icons/react';

export default function Sidebar({
  sites,
  selectedSite,
  setSelectedSite,
  dateRange,
  setDateRange,
  minTonality,
  setMinTonality,
  showPoints,
  setShowPoints
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
          <MapPin weight="duotone" /> Ubicación
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
          <Calendar weight="duotone" /> Rango de Fechas
        </label>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input 
            type="date" 
            className="custom-input" 
            value={dateRange.start}
            onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
          />
          <input 
            type="date" 
            className="custom-input" 
            value={dateRange.end}
            onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
          />
        </div>
      </div>

      <div className="control-group">
        <label className="control-label">
          <SlidersHorizontal weight="duotone" /> Tonalidad Mínima
        </label>
        <div className="slider-container">
          <input 
            type="range" 
            className="custom-slider" 
            min="0" 
            max="200" 
            step="1"
            value={minTonality}
            onChange={(e) => setMinTonality(Number(e.target.value))}
          />
          <span className="slider-value">{minTonality}</span>
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
