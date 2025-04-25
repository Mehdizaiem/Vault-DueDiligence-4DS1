import axios from 'axios';

// Interface for cryptocurrency data
export interface CryptoData {
  symbol: string;
  price: number;
  price_change_24h?: number;
  price_change_percentage_24h?: number;
}

// Interface for forecast data 
export interface ForecastData {
  timestamp: string;
  predicted_price: number;
  confidence_interval_lower: number;
  confidence_interval_upper: number;
  model_name?: string;
  model_type?: string;
  days_ahead?: number;
  average_uncertainty?: number;
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
 * Fetches the latest forecast for a cryptocurrency
 * @param symbol Cryptocurrency symbol
 * @returns Promise with forecast data
 */
export async function getLatestForecast(symbol: string = 'BTC_USD'): Promise<ForecastData | null> {
  try {
    const response = await axios.get(`/api/forecast/latest?symbol=${symbol}`);
    if (response.data.success && response.data.data) {
      return response.data.data;
    }
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
    return [];
  } catch (error) {
    console.error('Error fetching historical data:', error);
    return [];
  }
}

/**
 * Combines historical and forecast data for charting
 * @param historicalData Historical price data
 * @param forecastData Forecast data
 * @returns Combined data for charting
 */
export function combineDataForChart(historicalData: HistoricalData[], forecastData: ForecastData | null): ChartData[] {
  const chartData: ChartData[] = historicalData.map(item => ({
    date: item.date,
    price: item.price,
    predicted: undefined,
    lower: undefined,
    upper: undefined
  }));

  if (forecastData) {
    // Get the last date from historical data
    const lastHistoricalDate = new Date(historicalData[historicalData.length - 1]?.date || new Date());
    
    // For the forecast, we'll create 3 days of forecast data starting from the last historical date
    const forecastChartData: ChartData[] = [];
    
    for (let i = 1; i <= 3; i++) {
      const forecastDate = new Date(lastHistoricalDate);
      forecastDate.setDate(forecastDate.getDate() + i);
      
      forecastChartData.push({
        date: forecastDate.toISOString().split('T')[0],
        price: undefined,
        predicted: forecastData.predicted_price,
        lower: forecastData.confidence_interval_lower,
        upper: forecastData.confidence_interval_upper
      });
    }

    return [...chartData, ...forecastChartData];
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
 * Generate market insight based on forecast data
 * @param currentPrice Current price
 * @param forecastData Forecast data
 * @returns Market insight text
 */
export function generateInsight(currentPrice: number, forecastData: ForecastData): string {
  const predictedPrice = forecastData.predicted_price;
  const trend = calculateTrend(currentPrice, predictedPrice);
  const changePercent = ((predictedPrice - currentPrice) / currentPrice) * 100;
  const daysAhead = forecastData.days_ahead || 3;
  
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
}