import { fetchRiskProfiles } from "@/services/risk-service";

export async function GET() {
  try {
    const profiles = await fetchRiskProfiles();
    return Response.json({ success: true, data: profiles });
  } catch (error) {
    return Response.json({ success: false, error: "Failed to fetch risk data" }, { status: 500 });
  }
}
