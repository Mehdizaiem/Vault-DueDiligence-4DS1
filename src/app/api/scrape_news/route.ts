import { NextResponse } from "next/server";
import { exec } from "child_process";
import path from "path";

export async function GET() {
  return new Promise((resolve) => {
    const scriptPath = path.join(process.cwd(), "run_scraper.bat"); // or use .py/.sh
    exec(`python sentiment_store.py`, { cwd: path.join(process.cwd(), "Code") }, (err, stdout, stderr) => {
      if (err) {
        console.error("Scraping error:", stderr);
        return resolve(NextResponse.json({ error: stderr }, { status: 500 }));
      }
      console.log("Scraping done:", stdout);
      resolve(NextResponse.json({ message: "Scraping complete" }));
    });
  });
}
