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
  
  return new Date(Date.UTC(year, month, day, hour, min, sec));
}

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const [sites, setSites] = useState([]);
  const [selectedSite, setSelectedSite] = useState('');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [minTonality, setMinTonality] = useState(0);
  const [showPoints, setShowPoints] = useState(true);

  useEffect(() => {
    Papa.parse('/data/boat_detections_FINAL.csv', {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const parsedData = results.data.map(d => {
          let utcDate;
          try {
            utcDate = parseDatetimeFromFilename(d.file);
          } catch(e) {
            utcDate = new Date();
          }
          const chileDate = new Date(utcDate.getTime() - 3 * 3600 * 1000);
          
          return {
            ...d,
            datetime_utc: utcDate,
            datetime_chile: chileDate,
            date_chile: d.date_chile 
          };
        }).sort((a,b) => a.datetime_chile - b.datetime_chile);

        setData(parsedData);
        
        const uniqueSites = [...new Set(parsedData.map(d => d.site).filter(Boolean))].sort();
        setSites(uniqueSites);
        if (uniqueSites.length > 0) setSelectedSite(uniqueSites[0]);

        if (parsedData.length > 0) {
          const dates = parsedData.map(d => d.date_chile).filter(Boolean).sort();
          if (dates.length > 0) {
            setDateRange({
              start: dates[0],
              end: dates[dates.length - 1]
            });
          }
        }
        
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
      if (dateRange.start && d.date_chile < dateRange.start) return false;
      if (dateRange.end && d.date_chile > dateRange.end) return false;
      if (d.boat_tonality != null && d.boat_tonality < minTonality) return false;
      
      return true;
    });
  }, [data, selectedSite, dateRange, minTonality]);

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
          dateRange={dateRange}
          setDateRange={setDateRange}
          minTonality={minTonality}
          setMinTonality={setMinTonality}
          showPoints={showPoints}
          setShowPoints={setShowPoints}
        />
        
        <Charts 
          data={filteredData}
          showPoints={showPoints}
        />
      </div>
    </>
  );
}

export default App;
