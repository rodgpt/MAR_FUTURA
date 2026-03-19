import React, { useState, useEffect, useMemo } from 'react';
import Papa from 'papaparse';
import Sidebar from './components/Sidebar';
import Charts from './components/Charts';

function parseDatetimeFromFilename(filename) {
  const base = filename.replace(/\.[Ww][Aa][Vv][Ee]?$/, '');
  const year = parseInt(base.substring(0, 4));
  const month = parseInt(base.substring(4, 6)) - 1;
  const day = parseInt(base.substring(6, 8));
  const hour = parseInt(base.substring(9, 11));
  const min = parseInt(base.substring(11, 13));
  const sec = parseInt(base.substring(13, 15));
  
  // Create a UTC date representing the file time
  const fileUTC = new Date(Date.UTC(year, month, day, hour, min, sec));
  
  // Subtract 3 hours to get Chile time in UTC
  const chileMs = fileUTC.getTime() - (3 * 3600 * 1000);
  const chileUTC = new Date(chileMs);
  
  // Now reconstruct a browser-local Date treating the Chile time as local
  // so that date-fns formatting output the exact numbers as Chile time.
  return new Date(
    chileUTC.getUTCFullYear(),
    chileUTC.getUTCMonth(),
    chileUTC.getUTCDate(),
    chileUTC.getUTCHours(),
    chileUTC.getUTCMinutes(),
    chileUTC.getUTCSeconds()
  );
}

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const [sites, setSites] = useState([]);
  const [selectedSite, setSelectedSite] = useState('');
  const [showPoints, setShowPoints] = useState(true);
  
  // Drill-down and filters
  const [selectedDate, setSelectedDate] = useState(null);
  const [minDetections, setMinDetections] = useState(1);

  useEffect(() => {
    Papa.parse('/data/boat_detections_FINAL.csv', {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const parsedData = results.data.map(d => {
          let chileDate;
          try {
            chileDate = parseDatetimeFromFilename(d.file);
          } catch(e) {
            chileDate = new Date();
          }
          
          return {
            ...d,
            datetime_chile: chileDate,
            date_chile: d.date_chile 
          };
        }).sort((a,b) => a.datetime_chile - b.datetime_chile);

        setData(parsedData);
        
        const uniqueSites = [...new Set(parsedData.map(d => d.site).filter(Boolean))].sort();
        setSites(uniqueSites);
        if (uniqueSites.length > 0) setSelectedSite(uniqueSites[0]);
        
        setLoading(false);
      },
      error: (err) => {
        console.error("Error al cargar el CSV", err);
        setLoading(false);
      }
    });
  }, []);

  const filteredData = useMemo(() => {
    if (!data.length) return [];
    
    return data.filter(d => {
      if (d.site !== selectedSite) return false;
      return true;
    });
  }, [data, selectedSite]);
  
  // Whenever the site changes, reset the selected date to go back to the top-level view
  useEffect(() => {
    setSelectedDate(null);
  }, [selectedSite]);

  return (
    <>
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <div>Cargando datos de detección...</div>
        </div>
      )}
      
      <div className="dashboard-container">
        <Sidebar 
          sites={sites}
          selectedSite={selectedSite}
          setSelectedSite={setSelectedSite}
          showPoints={showPoints}
          setShowPoints={setShowPoints}
          minDetections={minDetections}
          setMinDetections={setMinDetections}
        />
        
        <Charts 
          data={filteredData}
          showPoints={showPoints}
          selectedDate={selectedDate}
          setSelectedDate={setSelectedDate}
          minDetections={minDetections}
        />
      </div>
    </>
  );
}

export default App;
