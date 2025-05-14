//src\app\api\risk-profiles\route.ts
import { fetchRiskProfiles } from "@/services/risk-service";

export async function GET() {
  try {
    const profiles = await fetchRiskProfiles();
    const validProfiles = profiles.filter(
      (p) => p.risk_score != null && p.risk_category != null
    );
    console.log(
      `🧮 Valid Profiles: ${validProfiles.length} / Total: ${profiles.length}`
    );
    return Response.json({ success: true, data: profiles });
  } catch (error) {
    return Response.json({ success: false, error: "Failed to fetch risk data" }, { status: 500 });
  }
}
