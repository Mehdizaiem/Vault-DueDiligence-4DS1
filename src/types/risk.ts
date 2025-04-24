export type RiskProfile = {
    symbol: string;
    risk_score: number;
    risk_category: string;
    analysis_timestamp: string;
    analysis_period_days: number;
    market_data_points: number;
    sentiment_data_points: number;
    risk_factors: string[];
    calculation_error?: string;
  };
  