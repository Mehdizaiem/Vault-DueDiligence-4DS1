// File: src/services/analytics-service.ts

// Define the type for the Analytics response
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
      sentiment_label: string;
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
  
export async function fetchAnalyticsData(): Promise<AnalyticsResponse> {
  const startTime = Date.now();
  const defaultResponse: AnalyticsResponse = {
    kpis: {},
    asset_distribution: [],
    portfolio_performance: [],
    recent_news: [],
    due_diligence: [],
  };

  try {
    const response = await fetch('/api/analytics', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Add a cache-busting parameter to avoid caching issues during development
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

    // Optionally store analytics fetch history
    try {
      await storeAnalyticsHistory(durationMs);
    } catch (historyError) {
      console.error('Error storing analytics history:', historyError);
      // Don't fail the main operation if history storage fails
    }

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

// Function to store analytics fetch history (optional)
async function storeAnalyticsHistory(durationMs: number): Promise<void> {
  try {
    const response = await fetch('/api/analytics/history', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        timestamp: new Date().toISOString(),
        durationMs,
      }),
    });

    if (!response.ok) {
      console.error('Failed to store analytics history:', await response.text());
    }
  } catch (error) {
    console.error('Error storing analytics history:', error);
    // Don't throw to avoid affecting the main flow
  }
}