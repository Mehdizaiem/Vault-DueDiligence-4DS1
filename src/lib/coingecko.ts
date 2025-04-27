interface CoinGeckoPrice {
  symbol: string;
  price: number;
  price_change_24h: number;
  price_change_percentage_24h: number;
}

export async function getLatestPrices(symbols: string[]): Promise<CoinGeckoPrice[]> {
  try {
    const response = await fetch(
      `https://api.coingecko.com/api/v3/simple/price?ids=${symbols.join(',')}&vs_currencies=usd&include_24hr_change=true`
    );
    
    if (!response.ok) {
      throw new Error('Failed to fetch prices from CoinGecko');
    }

    const data = await response.json();
    return Object.entries(data).map(([id, value]: [string, any]) => ({
      symbol: id.toUpperCase(),
      price: value.usd,
      price_change_24h: value.usd_24h_change,
      price_change_percentage_24h: value.usd_24h_change
    }));
  } catch (error) {
    console.error('Error fetching prices:', error);
    return [];
  }
} 