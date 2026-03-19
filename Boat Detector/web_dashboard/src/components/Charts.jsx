import React, { useMemo } from 'react';
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis, Brush
} from 'recharts';
import { format } from 'date-fns';
import { Waveform, WarningCircle } from '@phosphor-icons/react';

const CustomTooltip = ({ active, payload }) => {
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

export default function Charts({ data, showPoints }) {
  
  const timelineData = useMemo(() => {
    return data.map((d, index) => ({
      ...d,
      y: 1, 
      id: index,
      timestamp: d.datetime_chile.getTime()
    })).sort((a,b) => a.timestamp - b.timestamp);
  }, [data]);

  return (
    <div className="main-content">
      <div className="glass-panel chart-container" style={{ minHeight: '600px' }}>
        <div className="chart-header">
          <div>
            <div className="chart-title">
              <Waveform weight="bold" /> Línea de Tiempo de Detecciones
            </div>
            <div className="chart-subtitle">Ocurrencias de eventos en el período seleccionado, usa el selector (Brush) abajo para hacer zoom</div>
          </div>
        </div>
        
        {data.length === 0 ? (
          <div className="empty-state">
            <WarningCircle size={48} weight="duotone" />
            <h3>No se encontraron detecciones</h3>
            <p>Intente ajustar los filtros</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%" minHeight={500}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis 
                type="number" 
                dataKey="timestamp" 
                domain={['dataMin', 'dataMax']} 
                tickFormatter={(unixTime) => format(new Date(unixTime), 'MMM dd')}
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
              <Tooltip cursor={{stroke: 'rgba(3,8,47,0.1)', strokeWidth: 1}} content={<CustomTooltip />} />
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
                tickFormatter={(unixTime) => format(new Date(unixTime), 'MMM dd')} 
              />
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
