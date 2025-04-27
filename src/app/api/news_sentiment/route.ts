import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), "Sample_Data", "data_ingestion", "processed", "sentiment_20250423.csv"); // or dynamically use latest

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: "Sentiment data not found" }, { status: 404 });
    }

    const raw = fs.readFileSync(filePath, "utf-8");

    interface Record {
      title: string;
      source: string;
      date: string;
      url: string;
      content: string;
      sentiment_label: string;
      sentiment_score: string;
      aspect: string;
    }

    const records: Record[] = parse(raw, {
      columns: true,
      skip_empty_lines: true,
    });

    const articles = records.map((row, i) => ({
      id: `${i}`,
      title: row.title,
      source: row.source,
      date: row.date,
      url: row.url,
      content: row.content,
      sentiment_label: row.sentiment_label || "NEUTRAL",
      sentiment_score: isNaN(parseFloat(row.sentiment_score)) ? null : parseFloat(row.sentiment_score),
      aspect: row.aspect,
    }));

    return NextResponse.json({ articles });
  } catch (error) {
    console.error("❌ Failed to read CSV:", error);
    return NextResponse.json({ error: "CSV processing failed" }, { status: 500 });
  }
}
