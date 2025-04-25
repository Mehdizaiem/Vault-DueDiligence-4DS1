import axios from 'axios';

// Interface for cryptocurrency data
export interface CryptoData {
  symbol: string;
  price: number;
  price_change_24h?: number;
  price_change_percentage_24h?: number;
}

// Interface for forecast data from storage manager
export interface ForecastData {
  id?: string;
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
  error?: string; // Added to handle error responses
}

// Interface for historical price data
export interface HistoricalData {
  date: string;
  price: number;
}

// Interface for the combined historical and forecast data for chart
export interface ChartData {
  date: string;
  price?: number;
  predicted?: number;
  lower?: number;
  upper?: number;
}

/**
 * Fetches the latest forecast for a cryptocurrency from storage manager
 * @param symbol Cryptocurrency symbol
 * @returns Promise with forecast data
 */
export async function getLatestForecast(symbol: string = 'BTC_USD'): Promise<ForecastData | null> {
  try {
    // Get forecast data from storage manager via API
    const response = await axios.get(`/api/forecasts?symbol=${symbol}`);
    console.log('Forecast API response:', response.data);
    
    if (response.data.success && response.data.data && response.data.data.length > 0) {
      // Return the first forecast (most recent)
      return response.data.data[0];
    }
    
    // If API returns success: false, log the error but don't throw an exception
    if (!response.data.success) {
      console.warn('Forecast API returned error:', response.data.error);
      return null;
    }
    
    console.warn('No forecast data returned from API');
    return null;
  } catch (error) {
    console.error('Error fetching forecast data:', error);
    return null;
  }
}

/**
 * Fetches historical price data for a cryptocurrency
 * @param symbol Cryptocurrency symbol
 * @param days Number of days of historical data
 * @returns Promise with historical data
 */
export async function getHistoricalData(symbol: string = 'BTC_USD', days: number = 365): Promise<HistoricalData[]> {
  try {
    const response = await axios.get(`/api/historical?symbol=${symbol}&days=${days}`);
    
    if (response.data.success && response.data.data) {
      return response.data.data.map((item: any) => ({
        date: typeof item.date === 'string' ? item.date : new Date(item.date).toISOString().split('T')[0],
        price: item.price
      }));
    }
    
    console.warn('No historical data returned from API');
    return [];
  } catch (error) {
    console.error('Error fetching historical data:', error);
    return [];
  }
}

/**
 * Combines historical and forecast data for charting
 * @param historicalData Historical price data
 * @param forecastData Forecast data from storage manager
 * @returns Combined data for charting
 */
export function combineDataForChart(historicalData: HistoricalData[], forecastData: ForecastData | null): ChartData[] {
  if (!historicalData || historicalData.length === 0) return [];

  const chartData = historicalData.map(item => ({
    date: item.date,
    price: item.price,
    predicted: undefined,
    lower: undefined,
    upper: undefined
  }));

  // Only process forecast data if it exists and doesn't have an error
  if (forecastData && !forecastData.error && 
      Array.isArray(forecastData.forecast_dates) && 
      Array.isArray(forecastData.forecast_values)) {
    
    // Get valid forecast items
    const forecastItems: ChartData[] = [];
    
    for (let i = 0; i < forecastData.forecast_dates.length; i++) {
      forecastItems.push({
        date: forecastData.forecast_dates[i],
        price: undefined, // No historical price for forecast dates
        predicted: forecastData.forecast_values[i],
        lower: Array.isArray(forecastData.lower_bounds) ? forecastData.lower_bounds[i] : undefined,
        upper: Array.isArray(forecastData.upper_bounds) ? forecastData.upper_bounds[i] : undefined
      });
    }
    
    return [...chartData, ...forecastItems];
  }

  return chartData;
}

/**
 * Formats a price as currency
 * @param price Price to format
 * @returns Formatted price string
 */
export function formatPrice(price: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(price);
}

/**
 * Gets the color for a trend
 * @param trend Trend string (bullish, bearish, neutral)
 * @returns CSS color class
 */
export function getTrendColor(trend: string): string {
  if (!trend) return 'text-yellow-500';
  
  switch (trend.toLowerCase()) {
    case 'bullish':
    case 'strongly bullish':
      return 'text-green-500';
    case 'bearish':
    case 'strongly bearish':
      return 'text-red-500';
    default:
      return 'text-yellow-500';
  }
}

/**
 * Gets the background color for a trend
 * @param trend Trend string (bullish, bearish, neutral)
 * @returns CSS background color class
 */
export function getTrendBgColor(trend: string): string {
  if (!trend) return 'bg-yellow-500';
  
  switch (trend.toLowerCase()) {
    case 'bullish':
    case 'strongly bullish':
      return 'bg-green-500';
    case 'bearish':
    case 'strongly bearish':
      return 'bg-red-500';
    default:
      return 'bg-yellow-500';
  }
}

/**
 * Calculate trend based on current and predicted price
 * @param currentPrice Current price
 * @param predictedPrice Predicted price
 * @returns Trend string (bullish, bearish, neutral)
 */
export function calculateTrend(currentPrice: number, predictedPrice: number): string {
  const percentChange = ((predictedPrice - currentPrice) / currentPrice) * 100;
  
  if (percentChange > 5) return 'strongly bullish';
  if (percentChange > 1) return 'bullish';
  if (percentChange < -5) return 'strongly bearish';
  if (percentChange < -1) return 'bearish';
  return 'neutral';
}

/**
 * Generate market insight based on forecast data if it doesn't already have one
 * @param currentPrice Current price
 * @param forecastData Forecast data from storage manager
 * @returns Market insight text
 */
export function generateInsight(currentPrice: number, forecastData: ForecastData | null): string {
  if (!forecastData) {
    return "Forecast data is currently unavailable. Please check back later for market insights.";
  }
  
  // Check for error response
  if (forecastData.error) {
    return `Unable to retrieve forecast data: ${forecastData.error}`;
  }
  
  // If the forecast already has an insight, use it
  if (forecastData.insight && forecastData.insight.trim() !== '') {
    return forecastData.insight;
  }
  
  // Generate insight based on trend and forecast
  const trend = forecastData.trend || calculateTrend(currentPrice, forecastData.final_forecast);
  const changePercent = forecastData.change_pct || 
    ((forecastData.final_forecast - currentPrice) / currentPrice) * 100;
  const daysAhead = forecastData.days_ahead || 7;
  
  let insight = '';
  
  if (trend.includes('bullish')) {
    insight = `The forecast indicates a ${trend} trend for ${forecastData.symbol || 'BTC/USD'}, with a projected increase of ${changePercent.toFixed(2)}% over the next ${daysAhead} days.`;
  } else if (trend.includes('bearish')) {
    insight = `The forecast indicates a ${trend} trend for ${forecastData.symbol || 'BTC/USD'}, with a projected decrease of ${Math.abs(changePercent).toFixed(2)}% over the next ${daysAhead} days.`;
  } else {
    insight = `The forecast indicates a ${trend} trend for ${forecastData.symbol || 'BTC/USD'}, with minimal expected price movement (${changePercent.toFixed(2)}%) over the next ${daysAhead} days.`;
  }
  
  // Add probability information if available
  if (forecastData.probability_increase) {
    if (forecastData.probability_increase > 75) {
      insight += ` There is a strong probability (${forecastData.probability_increase.toFixed(1)}%) that the price will increase.`;
    } else if (forecastData.probability_increase > 60) {
      insight += ` There is a moderate probability (${forecastData.probability_increase.toFixed(1)}%) that the price will increase.`;
    } else if (forecastData.probability_increase < 25) {
      insight += ` There is a strong probability (${(100-forecastData.probability_increase).toFixed(1)}%) that the price will decrease.`;
    } else if (forecastData.probability_increase < 40) {
      insight += ` There is a moderate probability (${(100-forecastData.probability_increase).toFixed(1)}%) that the price will decrease.`;
    } else {
      insight += ` The price direction is uncertain with ${forecastData.probability_increase.toFixed(1)}% probability of increase.`;
    }
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
}