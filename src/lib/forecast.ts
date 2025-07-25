interface ForecastData {
  id: string;
  symbol: string;
  forecast_timestamp: string;
  model_name: string;
  model_type: string;
  days_ahead: number;
  current_price: number;
  forecast_dates: string[];
  forecast_values: number[];
  lower_bounds: number[];
  upper_bounds: number[];
  final_forecast: number;
  change_pct: number;
  trend: string;
  probability_increase: number;
  average_uncertainty: number;
  insight: string;
}

export async function getForecastData(symbol: string): Promise<ForecastData | null> {
  try {
    const response = await fetch(`/api/forecasts?symbol=${symbol}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch forecast data');
    }

    const result = await response.json();
    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch forecast data');
    }

    return result.data;
  } catch (error) {
    console.error('Error fetching forecast data:', error);
    return null;
  }
} 