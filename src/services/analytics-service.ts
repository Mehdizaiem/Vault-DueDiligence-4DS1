// src/services/analytics-service.ts

export interface AnalyticsResponse {
  kpis: {
    market_cap?: { value: string; change: string; trend: string };
    asset_count?: { value: string; change: string; trend: string };
    price_change?: { value: string; change: string; trend: string };
    market_sentiment?: { value: string; change: string; trend: string };
  };
  asset_distribution: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  portfolio_performance: Array<{
    symbol: string;
    timestamp: string;
    change_pct: number;
  }>;
  recent_news: Array<{
    title: string;
    source: string;
    date: string;
    sentiment_score: number;
    sentiment_color: string;
    related_assets: string[];
    url: string;
  }>;
  due_diligence: Array<{
    title: string;
    document_type: string;
    source: string;
    keywords: string[];
    icon: string;
    id: string;
  }>;
  error?: string;
}

export interface AnalyticsFilters {
  dateRange: string;
  symbols: string[];
}

export async function fetchAnalyticsData(filters: AnalyticsFilters): Promise<AnalyticsResponse> {
  const startTime = Date.now();
  const defaultResponse: AnalyticsResponse = {
    kpis: {},
    asset_distribution: [],
    portfolio_performance: [],
    recent_news: [],
    due_diligence: [],
  };

  try {
    // Build query parameters
    const params = new URLSearchParams();
    params.append('dateRange', filters.dateRange);
    
    // Only add symbol filters if they're not 'All'
    filters.symbols.forEach(symbol => {
      if (symbol !== 'All') {
        params.append('symbols', symbol);
      }
    });

    console.log('Fetching analytics with filters:', filters);
    
    const response = await fetch(`/api/analytics?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('API response not OK:', errorData);
      return {
        ...defaultResponse,
        error: errorData.error || 'Failed to fetch analytics data'
      };
    }

    const result = await response.json();
    const durationMs = Date.now() - startTime;

    // Log performance metrics
    console.log(`Analytics data fetched in ${durationMs}ms`);

    // If result has an error field, include it in the response
    if (result.error) {
      console.warn('API returned an error:', result.error);
    }

    return {
      kpis: result.kpis || {},
      asset_distribution: result.asset_distribution || [],
      portfolio_performance: result.portfolio_performance || [],
      recent_news: result.recent_news || [],
      due_diligence: result.due_diligence || [],
      error: result.error,
    };
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    return {
      ...defaultResponse,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}