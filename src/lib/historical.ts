interface HistoricalData {
  date: string;
  price: number;
  volume: number;
  market_cap: number;
}

export async function getHistoricalData(symbol: string): Promise<HistoricalData[]> {
  try {
    const response = await fetch(
      `https://api.coingecko.com/api/v3/coins/${symbol.toLowerCase()}/market_chart?vs_currency=usd&days=30&interval=daily`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch historical data from CoinGecko');
    }

    const data = await response.json();
    return data.prices.map(([timestamp, price]: [number, number], index: number) => ({
      date: new Date(timestamp).toISOString().split('T')[0],
      price,
      volume: data.total_volumes[index][1],
      market_cap: data.market_caps[index][1]
    }));
  } catch (error) {
    console.error('Error fetching historical data:', error);
    return [];
  }
} 