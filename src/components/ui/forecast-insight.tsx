import { Calendar, ExternalLink, ShieldAlert } from 'lucide-react';

interface ForecastInsightProps {
  currentPrice: number;
  forecastData: {
    trend: string;
    change_pct: number;
    probability_increase: number;
    average_uncertainty: number;
    model_name: string;
    model_type: string;
    days_ahead: number;
    final_forecast: number;
    forecast_timestamp: string;
    insight: string;
    lower_bounds?: number[];
    upper_bounds?: number[];
  } | null;
}

export default function ForecastInsight({ currentPrice, forecastData }: ForecastInsightProps) {
  if (!forecastData) {
    return null;
  }

  // Format date for display
  const formatDate = (dateStr: string) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'long', 
      day: 'numeric', 
      year: 'numeric',
      hour: 'numeric',
      minute: 'numeric'
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

  // Get confidence interval range
  const getConfidenceRange = () => {
    if (!forecastData.lower_bounds || !forecastData.upper_bounds) return null;
    
    const lastLower = forecastData.lower_bounds[forecastData.lower_bounds.length - 1];
    const lastUpper = forecastData.upper_bounds[forecastData.upper_bounds.length - 1];
    
    return {
      lower: lastLower,
      upper: lastUpper
    };
  };

  const confidenceRange = getConfidenceRange();

  return (
    <div className="space-y-4">
      <p className="text-gray-700">
        {forecastData.insight}
      </p>
      <div className="grid gap-6 md:grid-cols-3 mt-6 pt-6 border-t border-gray-100">
        <div className="flex items-start space-x-2">
          <Calendar className="h-4 w-4 text-gray-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-gray-500">Forecast Details</h4>
            <p className="text-sm font-medium mt-1">
              {forecastData.days_ahead} days ahead
              <br />
              <span className="text-gray-500 text-xs">
                Generated: {formatDate(forecastData.forecast_timestamp)}
              </span>
            </p>
            {confidenceRange && (
              <p className="text-xs text-gray-500 mt-1">
                Range: {formatPrice(confidenceRange.lower)} - {formatPrice(confidenceRange.upper)}
              </p>
            )}
          </div>
        </div>
        
        <div className="flex items-start space-x-2">
          <ExternalLink className="h-4 w-4 text-gray-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-gray-500">Model</h4>
            <p className="text-sm font-medium mt-1">
              {forecastData.model_name}
              <br />
              <span className="text-gray-500 text-xs">
                Type: {forecastData.model_type}
              </span>
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Probability of Increase: {forecastData.probability_increase.toFixed(1)}%
            </p>
          </div>
        </div>
        
        <div className="flex items-start space-x-2">
          <ShieldAlert className="h-4 w-4 text-gray-400 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-gray-500">Uncertainty</h4>
            <p className="text-sm font-medium mt-1">
              ±{forecastData.average_uncertainty.toFixed(1)}%
              <br />
              <span className={`text-xs ${
                forecastData.average_uncertainty > 20 ? 'text-red-500' :
                forecastData.average_uncertainty > 10 ? 'text-yellow-500' :
                'text-green-500'
              }`}>
                {forecastData.average_uncertainty > 20 ? 'High' :
                 forecastData.average_uncertainty > 10 ? 'Moderate' :
                 'Low'} Uncertainty
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}