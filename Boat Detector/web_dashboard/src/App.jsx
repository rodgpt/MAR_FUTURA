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
  const [selectedSite, setSelectedSite] = useState('Todos');
  const [showPoints, setShowPoints] = useState(true);
  
  // Drill-down and filters
  const [selectedDate, setSelectedDate] = useState(null);
  const [minDetections, setMinDetections] = useState(1);
  const [startHour, setStartHour] = useState(0);
  const [endHour, setEndHour] = useState(23);

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
        
        // Group by base site name
        const uniqueGroups = [...new Set(parsedData.map(d => {
          if (!d.site) return "";
          return d.site.replace(/\s*\d+$/, '').trim();
        }).filter(Boolean))].sort();
        
        setSites(["Todos", ...uniqueGroups]);
        
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
      // 1. Filter by location group
      const dGroup = d.site ? d.site.replace(/\s*\d+$/, '').trim() : "";
      if (selectedSite !== "Todos" && dGroup !== selectedSite) return false;
      
      // 2. Filter by hour range
      const hour = d.datetime_chile.getHours();
      if (startHour <= endHour) {
         // Normal range, e.g. 08 to 18
         if (hour < startHour || hour > endHour) return false;
      } else {
         // Overnight wrap-around range, e.g. 22 to 06
         if (hour < startHour && hour > endHour) return false;
      }

      return true;
    });
  }, [data, selectedSite, startHour, endHour]);
  
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
          startHour={startHour}
          setStartHour={setStartHour}
          endHour={endHour}
          setEndHour={setEndHour}
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
