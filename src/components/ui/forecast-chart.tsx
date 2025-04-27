import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area } from 'recharts';
import { ChartData } from '@/lib/forecast-manager';

interface ForecastChartProps {
  data: ChartData[];
  timeFrame: '90d' | '180d' | '365d';
}

export default function ForecastChart({ data, timeFrame }: ForecastChartProps) {
  // Function to filter chart data based on selected time frame
  const getFilteredData = () => {
    if (data.length === 0) return [];
    
    const days = timeFrame === '90d' ? 90 : timeFrame === '180d' ? 180 : 365;
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);
    
    return data.filter(item => {
      const itemDate = new Date(item.date);
      return itemDate >= cutoffDate;
    });
  };

  // Format date for display
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    
    if (timeFrame === '90d') {
      // Format as "Jan 15" for 90-day view
      return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' }).format(date);
    } else {
      // Format as "Jan 2023" for longer views
      return new Intl.DateTimeFormat('en-US', { month: 'short', year: 'numeric' }).format(date);
    }
  };

  // Format date for tooltip
  const formatTooltipDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'long', 
      day: 'numeric', 
      year: 'numeric' 
    }).format(date);
  };

  const filteredData = getFilteredData();

  // Find the range for Y-axis
  const yDomain = filteredData.reduce((acc, item) => {
    const values = [
      item.price,
      item.predicted,
      item.lower,
      item.upper
    ].filter(v => v !== undefined && v !== null) as number[];

    return {
      min: Math.min(acc.min, ...values),
      max: Math.max(acc.max, ...values)
    };
  }, { min: Infinity, max: -Infinity });

  // Add some padding to the Y-axis range
  const yPadding = (yDomain.max - yDomain.min) * 0.1;
  
  return (
    <div className="h-[400px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={filteredData} 
          margin={{ top: 10, right: 30, left: 10, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.6} />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatDate}
            tick={{ fontSize: 12 }}
            padding={{ left: 10, right: 10 }}
          />
          <YAxis 
            tickFormatter={(value) => `$${Number(value).toLocaleString()}`}
            domain={[yDomain.min - yPadding, yDomain.max + yPadding]}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            labelFormatter={(value) => formatTooltipDate(value)}
            formatter={(value: any, name: string) => {
              if (value === null || value === undefined) return ['-', name];
              const formattedValue = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
              }).format(value);
              
              const nameMap: {[key: string]: string} = {
                'price': 'Historical Price',
                'predicted': 'Forecast Price',
                'lower': 'Lower Bound',
                'upper': 'Upper Bound'
              };
              
              return [formattedValue, nameMap[name] || name];
            }}
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #e2e8f0',
              borderRadius: '0.5rem',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              padding: '8px 12px',
              fontSize: '12px'
            }}
          />
          <Legend 
            iconType="line"
            align="center"
            wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
            formatter={(value) => {
              const nameMap: {[key: string]: string} = {
                'price': 'Historical Price',
                'predicted': 'Forecast Price',
                'confidence': 'Confidence Interval'
              };
              return nameMap[value] || value;
            }}
          />
          
          {/* Confidence Interval Area */}
          <Area
            name="confidence"
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="#10B981"
            fillOpacity={0.1}
            activeDot={false}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#10B981"
            fillOpacity={0.1}
            activeDot={false}
          />
          
          {/* Historical Price Line */}
          <Line 
            name="price"
            type="monotone" 
            dataKey="price" 
            stroke="#4F46E5" 
            strokeWidth={2}
            dot={false}
            isAnimationActive={true}
            activeDot={{ r: 6, stroke: '#4F46E5', strokeWidth: 2, fill: 'white' }}
            connectNulls={true}
          />
          
          {/* Forecast Line */}
          <Line 
            name="predicted"
            type="monotone" 
            dataKey="predicted" 
            stroke="#10B981" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
            isAnimationActive={true}
            activeDot={{ r: 6, stroke: '#10B981', strokeWidth: 2, fill: 'white' }}
            connectNulls={true}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}