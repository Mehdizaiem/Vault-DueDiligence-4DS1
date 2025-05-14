//src\services\risk-service.ts
import { RiskProfile } from "@/types/risk";

export async function fetchRiskProfiles(): Promise<RiskProfile[]> {
    const query = `
    {
      Get {
        RiskProfiles(
          where: {
            operator: And,
            operands: [
              { path: ["symbol"], operator: NotEqual, valueText: "null" },
              { path: ["risk_score"], operator: GreaterThan, valueNumber: 0 },
              { path: ["risk_category"], operator: NotEqual, valueText: "null" }
            ]
          },
          limit: 1000
        ) {
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
    }`;
    

  const res = await fetch("http://localhost:9090/v1/graphql", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  const json = await res.json();
  console.log("Weaviate returned:", json?.data?.Get?.RiskProfiles?.length);
  //console.log("Sample:", json?.data?.Get?.RiskProfiles?.slice(0, 3));
  console.log(
    "Sample RiskProfiles entries:",
    json.data?.Get?.RiskProfiles?.slice(0, 10).map((p: RiskProfile) => ({
      symbol: p.symbol,
      risk_score: p.risk_score,
      risk_category: p.risk_category
    }))
  );
  const allProfiles = json.data?.Get?.RiskProfiles || [];

  const validProfiles = allProfiles.filter((p: any) => p.risk_score != null && p.risk_category != null);

  console.log(`Valid Profiles: ${validProfiles.length} / Total: ${allProfiles.length}`);

  return allProfiles;;
}