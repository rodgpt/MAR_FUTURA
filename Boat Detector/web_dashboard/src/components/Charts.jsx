import React, { useMemo } from 'react';
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ZAxis, Brush,
  LineChart, Line
} from 'recharts';
import { format, parseISO, eachDayOfInterval } from 'date-fns';
import { Waveform, WarningCircle, ChartLineUp, ArrowLeft } from '@phosphor-icons/react';

const ScatterTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="recharts-default-tooltip">
        <p style={{ margin: 0, fontWeight: 600, color: '#03082F' }}>
          {format(data.datetime_chile, 'MMM dd, yyyy HH:mm:ss')}
        </p>
        <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem' }}>
          Archivo: {data.file}
        </p>
        {data.boat_tonality && (
          <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem' }}>
            Tonalidad: <span style={{ color: '#3DAECA', fontWeight: 600 }}>{data.boat_tonality.toFixed(2)}</span>
          </p>
        )}
      </div>
    );
  }
  return null;
};

const DailyTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="recharts-default-tooltip">
        <p style={{ margin: 0, fontWeight: 600, color: '#03082F' }}>
          {format(parseISO(data.date), 'MMMM dd, yyyy')}
        </p>
        <p style={{ margin: '4px 0 0 0', fontSize: '0.9rem' }}>
          Detecciones: <span style={{ color: '#3DAECA', fontWeight: 600 }}>{data.count}</span>
        </p>
        {data.count > 0 ? (
          <p style={{ margin: '4px 0 0 0', fontSize: '0.75rem', color: '#888' }}>
            Clic para ver el detalle de los puntos
          </p>
        ) : (
          <p style={{ margin: '4px 0 0 0', fontSize: '0.75rem', color: '#888' }}>
            No cumple el mínimo de detecciones
          </p>
        )}
      </div>
    );
  }
  return null;
}

export default function Charts({ data, showPoints, selectedDate, setSelectedDate, minDetections }) {
  
  const dailyData = useMemo(() => {
    if (!data.length) return [];
    const counts = {};
    data.forEach(d => {
      counts[d.date_chile] = (counts[d.date_chile] || 0) + 1;
    });

    // Create a continuous timeline from the first to the last date
    const startStr = data[0].date_chile;
    const endStr = data[data.length - 1].date_chile;
    
    // Safety check just in case date_chile is malformed
    let allDays = [];
    try {
      allDays = eachDayOfInterval({
        start: parseISO(startStr),
        end: parseISO(endStr)
      });
    } catch (e) {
      // Fallback if parsing fails for some reason
      return Object.keys(counts).sort().map(date => ({
        date,
        count: counts[date] >= minDetections ? counts[date] : 0
      }));
    }

    return allDays.map(dateObj => {
      const dateStr = format(dateObj, 'yyyy-MM-dd');
      const count = counts[dateStr] || 0;
      return {
        date: dateStr,
        count: count >= minDetections ? count : 0
      };
    });
  }, [data, minDetections]);

  const timelineData = useMemo(() => {
    if (!selectedDate) return [];
    return data.filter(d => d.date_chile === selectedDate)
      .map((d, index) => ({
        ...d,
        y: 1, 
        id: index,
        timestamp: d.datetime_chile.getTime()
      })).sort((a,b) => a.timestamp - b.timestamp);
  }, [data, selectedDate]);

  return (
    <div className="main-content">
      <div className="glass-panel chart-container" style={{ minHeight: '600px' }}>
        <div className="chart-header">
          <div>
            <div className="chart-title">
              {selectedDate ? (
                <>
                  <Waveform weight="bold" /> Detalles del Día: {format(parseISO(selectedDate), 'MMM dd, yyyy')}
                </>
              ) : (
                <>
                  <ChartLineUp weight="bold" /> Resumen Diario de Detecciones
                </>
              )}
            </div>
            <div className="chart-subtitle">
              {selectedDate 
                ? "Puntos de detección precisos para el día seleccionado" 
                : "Haz clic en un punto para hacer zoom y ver los detalles de ese día"}
            </div>
          </div>
          {selectedDate && (
             <button className="btn-back" onClick={() => setSelectedDate(null)}>
               <ArrowLeft weight="bold" /> Volver al Resumen
             </button>
          )}
        </div>
        
        {data.length === 0 ? (
          <div className="empty-state">
            <WarningCircle size={48} weight="duotone" />
            <h3>No se encontraron detecciones</h3>
            <p>Intente elegir otra ubicación</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%" minHeight={500}>
            {!selectedDate ? (
              <LineChart 
                data={dailyData} 
                margin={{ top: 20, right: 20, bottom: 20, left: 0 }}
                onClick={(state) => {
                  if (state && state.activePayload && state.activePayload.length) {
                     const payload = state.activePayload[0].payload;
                     if (payload.count > 0) {
                        setSelectedDate(payload.date);
                     }
                  } else if (state && state.activeLabel) {
                     // Check if activeLabel has count > 0 in dailyData
                     const found = dailyData.find(d => d.date === state.activeLabel);
                     if (found && found.count > 0) {
                        setSelectedDate(state.activeLabel);
                     }
                  }
                }}
                style={{ cursor: 'pointer' }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(date) => {
                    try {
                      return format(parseISO(date), 'MMM dd');
                    } catch (e) {
                      return date;
                    }
                  }}
                  stroke="var(--text-secondary)"
                  tick={{ fill: 'var(--text-secondary)' }}
                />
                <YAxis 
                  stroke="var(--text-secondary)"
                  tick={{ fill: 'var(--text-secondary)' }}
                  allowDecimals={false}
                />
                <RechartsTooltip cursor={{ fill: 'rgba(3,8,47,0.05)' }} content={<DailyTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="count" 
                  stroke="#3DAECA" 
                  strokeWidth={3}
                  dot={{ 
                    fill: '#03082F', 
                    strokeWidth: 2, 
                    r: 5,
                    cursor: 'pointer',
                    onClick: (e, payload) => {
                      if (payload && payload.payload && payload.payload.count > 0) {
                         setSelectedDate(payload.payload.date);
                      }
                    }
                  }}
                  activeDot={{ 
                    r: 8, 
                    fill: '#3DAECA', 
                    stroke: '#03082F', 
                    strokeWidth: 2,
                    cursor: 'pointer',
                    onClick: (e, payload) => {
                      if (payload && payload.payload && payload.payload.count > 0) {
                         setSelectedDate(payload.payload.date);
                      }
                    }
                  }}
                />
              </LineChart>
            ) : (
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis 
                  type="number" 
                  dataKey="timestamp" 
                  domain={['dataMin', 'dataMax']} 
                  tickFormatter={(unixTime) => format(new Date(unixTime), 'HH:mm')}
                  stroke="var(--text-secondary)"
                  tick={{ fill: 'var(--text-secondary)' }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  hide={true} 
                  domain={[0, 2]} 
                />
                <ZAxis 
                  type="number" 
                  dataKey="boat_tonality" 
                  range={[40, 150]} 
                />
                <RechartsTooltip cursor={{stroke: 'rgba(3,8,47,0.1)', strokeWidth: 1}} content={<ScatterTooltip />} />
                <Scatter 
                  data={timelineData} 
                  fill="url(#colorTonality)" 
                  opacity={0.8}
                  shape={showPoints ? "circle" : "cross"}
                >
                </Scatter>
                <defs>
                  <linearGradient id="colorTonality" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#3DAECA" />
                    <stop offset="100%" stopColor="#03082F" />
                  </linearGradient>
                </defs>
                <Brush 
                  dataKey="timestamp" 
                  height={30} 
                  stroke="#3DAECA" 
                  tickFormatter={(unixTime) => format(new Date(unixTime), 'HH:mm')} 
                />
              </ScatterChart>
            )}
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
