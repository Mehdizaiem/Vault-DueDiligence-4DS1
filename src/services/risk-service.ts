import { RiskProfile } from "@/types/risk";

export async function fetchRiskProfiles(): Promise<RiskProfile[]> {
    const query = `
    {
      Get {
        RiskProfiles(limit: 1000) {
          symbol
          risk_score
          risk_category
          analysis_timestamp
          analysis_period_days
          market_data_points
          sentiment_data_points
          risk_factors
          calculation_error
        }
      }
    }
    `;
    

  const res = await fetch("http://localhost:9090/v1/graphql", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  const json = await res.json();
  console.log("Weaviate returned:", json?.data?.Get?.RiskProfiles?.length); // <--- Add this
  return json.data?.Get?.RiskProfiles || [];
}
