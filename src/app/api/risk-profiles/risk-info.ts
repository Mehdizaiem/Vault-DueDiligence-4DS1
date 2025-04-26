// File: src/pages/api/risk-info.ts
import type { NextApiRequest, NextApiResponse } from "next";
import { getWeaviateClient } from "@/lib/weaviate-client";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { document_id } = req.query;
  if (!document_id || typeof document_id !== "string") {
    return res.status(400).json({ error: "Missing or invalid document_id" });
  }

  try {
    const client = getWeaviateClient();

    const result = await client.graphql
      .get()
      .withClassName("User_doc_risk")
      .withFields("document_id risk_score risk_category risk_factors")
      .withWhere({
        path: ["document_id"],
        operator: "Equal",
        valueText: document_id,
      })
      .withLimit(1)
      .do();

    const doc = result.data?.Get?.user_doc_risk?.[0];

    if (!doc) {
      return res.status(404).json({ error: "Risk data not found" });
    }

    return res.status(200).json({
      risk_score: doc.risk_score,
      risk_category: doc.risk_category,
      risk_factors: doc.risk_factors,
    });

  } catch (err) {
    console.error("Error fetching risk info:", err);
    return res.status(500).json({ error: "Failed to fetch risk info" });
  }
}
