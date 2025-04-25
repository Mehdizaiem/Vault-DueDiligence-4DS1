import { ForecastData } from '@/lib/forecast-manager';
import { Calendar, ExternalLink, ShieldAlert, Info } from 'lucide-react';

interface ForecastInsightProps {
  currentPrice: number;
  forecastData: ForecastData | null;
}

export default function ForecastInsight({ currentPrice, forecastData }: ForecastInsightProps) {
  if (!forecastData) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg text-center">
        <p className="text-gray-500">No forecast data available at this time.</p>
      </div>
    );
  }

  // Generate insight text based on forecast data
  const generateInsight = () => {
    const predictedPrice = forecastData.predicted_price;
    const changePercent = ((predictedPrice - currentPrice) / currentPrice) * 100;
    const daysAhead = forecastData.days_ahead || 3;
    
    let trend;
    if (changePercent > 5) trend = 'strongly bullish';
    else if (changePercent > 1) trend = 'bullish';
    else if (changePercent < -5) trend = 'strongly bearish';
    else if (changePercent < -1) trend = 'bearish';
    else trend = 'neutral';
    
    let insight = '';
    
    if (trend.includes('bullish')) {
      insight = `The forecast indicates a ${trend} trend for BTC/USD, with a projected increase of ${changePercent.toFixed(2)}% over the next ${daysAhead} days.`;
    } else if (trend.includes('bearish')) {
      insight = `The forecast indicates a ${trend} trend for BTC/USD, with a projected decrease of ${Math.abs(changePercent).toFixed(2)}% over the next ${daysAhead} days.`;
    } else {
      insight = `The forecast indicates a ${trend} trend for BTC/USD, with minimal expected price movement (${changePercent.toFixed(2)}%) over the next ${daysAhead} days.`;
    }
    
    // Add uncertainty information
    if (forecastData.average_uncertainty) {
      if (forecastData.average_uncertainty > 20) {
        insight += ` The forecast shows high uncertainty (±${forecastData.average_uncertainty.toFixed(1)}%), suggesting caution in decision-making.`;
      } else if (forecastData.average_uncertainty > 10) {
        insight += ` The forecast shows moderate uncertainty (±${forecastData.average_uncertainty.toFixed(1)}%).`;
      } else {
        insight += ` The forecast shows relatively low uncertainty (±${forecastData.average_uncertainty.toFixed(1)}%).`;
      }
    }
    
    return insight;
  };

  // Format date for display
  const formatDate = (dateStr: string) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'long', 
      day: 'numeric', 
      year: 'numeric' 
    }).format(date);
  };

  // Format price as currency
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  return (
    <div className="space-y-4">
      <p className="text-gray-700">
        {generateInsight()}
      </p>
      <div className="grid gap-6 md:grid-cols-3 mt-6 pt-6 border-t border-gray-100">
        <div className="flex items-start space-x-2">
          <Calendar className="h-4 w-4 text-gray-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-gray-500">Forecast Range</h4>
            <p className="text-sm font-medium mt-1">
              {formatPrice(forecastData.confidence_interval_lower)} - {formatPrice(forecastData.confidence_interval_upper)}
            </p>
          </div>
        </div>
        
        <div className="flex items-start space-x-2">
          <ExternalLink className="h-4 w-4 text-gray-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-gray-500">Model</h4>
            <p className="text-sm font-medium mt-1">
              {forecastData.model_name || 'Chronos'} 
              {forecastData.model_type ? ` (${forecastData.model_type})` : ''}
            </p>
          </div>
        </div>
        
        <div className="flex items-start space-x-2">
          <ShieldAlert className="h-4 w-4 text-gray-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-gray-500">Uncertainty</h4>
            <p className="text-sm font-medium mt-1">
              ±{(forecastData.average_uncertainty || 0).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}