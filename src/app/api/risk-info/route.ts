// src/app/api/risk-info/route.ts
import { NextResponse } from 'next/server';
import {getWeaviateClient} from '@/lib/weaviate-client';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const documentId = searchParams.get("document_id");

  if (!documentId) {
    return NextResponse.json({ error: "Missing document_id" }, { status: 400 });
  }

  try {
    const client = getWeaviateClient();
    const result = await client.graphql.get()
      .withClassName("user_doc_risk")
      .withFields("document_id title risk_score risk_category risk_factors")
      .withWhere({
        path: ["document_id"],
        operator: "Equal",
        valueText: documentId,
      })
      .do();

    const risk = result?.data?.Get?.user_doc_risk?.[0];

    if (!risk) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    return NextResponse.json(risk);

  } catch (e) {
    console.error("Risk API error", e);
    return NextResponse.json({ error: "Failed to fetch risk info" }, { status: 500 });
  }
}
