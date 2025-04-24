// File: src/app/(dashboard)/(routes)/dashboard/risk/page.tsx

"use client";
import { useEffect, useState } from "react";
import { RiskProfile } from "@/types/risk";
import { Bar, Pie, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  TimeScale,
  ChartOptions
} from "chart.js";
import { saveAs } from "file-saver";
import "chartjs-adapter-date-fns";

ChartJS.register(
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
  TimeScale
);

const riskColors: Record<
  "Very Low" | "Low" | "Moderate" | "High" | "Very High" | "Error" | "Undetermined",
  string
> = {
  "Very Low": "bg-green-100 text-green-800",
  Low: "bg-lime-100 text-lime-800",
  Moderate: "bg-yellow-100 text-yellow-800",
  High: "bg-orange-100 text-orange-800",
  "Very High": "bg-red-100 text-red-800",
  Error: "bg-gray-200 text-gray-700",
  Undetermined: "bg-gray-100 text-gray-700"
};

export default function RiskDashboardPage() {
  const [allProfiles, setAllProfiles] = useState<RiskProfile[]>([]);
  const [error, setError] = useState("");
  const [filterText, setFilterText] = useState("");
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({ start: "", end: "" });

  useEffect(() => {
    fetch("/api/risk-profiles")
      .then((res) => res.json())
      .then((data) => {
        if (data.success) setAllProfiles(data.data);
        else setError("Failed to load risk data.");
      })
      .catch(() => setError("Error connecting to API."));
  }, []);

  const isWithinRange = (date: string) => {
    if (!dateRange.start && !dateRange.end) return true;
    const ts = new Date(date).getTime();
    const startTs = dateRange.start ? new Date(dateRange.start).getTime() : -Infinity;
    const endTs = dateRange.end ? new Date(dateRange.end).getTime() : Infinity;
    return ts >= startTs && ts <= endTs;
  };

  const filtered = allProfiles.filter(
    (p) =>
      p.risk_score != null &&
      p.risk_score >= 0 &&
      isWithinRange(p.analysis_timestamp) &&
      (p.symbol.toLowerCase().includes(filterText.toLowerCase()) ||
        p.risk_category.toLowerCase().includes(filterText.toLowerCase()))
  );

  const highest = [...filtered].sort((a, b) => b.risk_score - a.risk_score)[0];
  const lowest = [...filtered].sort((a, b) => a.risk_score - b.risk_score)[0];
  const average = filtered.length > 0 ? filtered.reduce((sum, p) => sum + (p.risk_score || 0), 0) / filtered.length : 0;

  const barData = {
    labels: filtered.map((p) => p.symbol),
    datasets: [
      {
        label: "Risk Score",
        data: filtered.map((p) => p.risk_score),
        backgroundColor: "#3b82f6"
      }
    ]
  };

  const pieCounts = filtered.reduce((acc, curr) => {
    acc[curr.risk_category] = (acc[curr.risk_category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const pieData = {
    labels: Object.keys(pieCounts),
    datasets: [
      {
        label: "Risk Categories",
        data: Object.values(pieCounts),
        backgroundColor: ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
      }
    ]
  };

  const exportCSV = () => {
    const csvRows = [
      ["Symbol", "Score", "Category", "Timestamp"],
      ...filtered.map((p) => [
        p.symbol,
        p.risk_score != null ? p.risk_score.toFixed(2) : "N/A",
        p.risk_category,
        new Date(p.analysis_timestamp).toLocaleString("en-US")
      ])
    ];
    const csvContent = csvRows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    saveAs(blob, "risk_profiles.csv");
  };

  const riskTimeSeries = selectedSymbol
    ? allProfiles
        .filter((p) => p.symbol === selectedSymbol && p.risk_score !== -1)
        .sort((a, b) => new Date(a.analysis_timestamp).getTime() - new Date(b.analysis_timestamp).getTime())
    : [];

  const lineData = {
    labels: riskTimeSeries.map((p) => new Date(p.analysis_timestamp)),
    datasets: [
      {
        label: `${selectedSymbol} Risk Over Time`,
        data: riskTimeSeries.map((p) => p.risk_score),
        borderColor: "#3b82f6",
        fill: false
      }
    ]
  };

  const lineOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top"
      }
    },
    scales: {
      x: {
        type: "time",
        time: {
          unit: "day"
        },
        title: {
          display: true,
          text: "Date"
        }
      },
      y: {
        title: {
          display: true,
          text: "Risk Score"
        },
        suggestedMin: 0,
        suggestedMax: 100
      }
    }
  };

  return (
    <div className="p-8 space-y-6">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">📊 Risk Analysis Dashboard</h1>

      {error && <p className="text-red-600">{error}</p>}

      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <input
          type="text"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          placeholder="🔍 Filter by symbol or category..."
          className="border p-2 rounded w-full max-w-xs text-sm"
        />
        <div className="flex gap-2">
          <input
            type="date"
            value={dateRange.start}
            onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
            className="border p-2 rounded text-sm"
          />
          <input
            type="date"
            value={dateRange.end}
            onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
            className="border p-2 rounded text-sm"
          />
        </div>
        <button
          onClick={exportCSV}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 text-sm rounded"
        >
          📁 Export CSV
        </button>
      </div>


      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white shadow rounded p-4">
          <h2 className="text-sm font-semibold text-gray-600">Highest Risk</h2>
          <p className="text-xl font-bold text-red-600">{highest?.symbol} ({highest?.risk_score?.toFixed(2)})</p>
          <span className={`text-xs font-medium px-2 py-1 rounded ${riskColors[highest?.risk_category as keyof typeof riskColors]}`}>{highest?.risk_category}</span>
        </div>
        <div className="bg-white shadow rounded p-4">
          <h2 className="text-sm font-semibold text-gray-600">Lowest Risk</h2>
          <p className="text-xl font-bold text-green-600">{lowest?.symbol} ({lowest?.risk_score?.toFixed(2)})</p>
          <span className={`text-xs font-medium px-2 py-1 rounded ${riskColors[lowest?.risk_category as keyof typeof riskColors]}`}>{lowest?.risk_category}</span>
        </div>
        <div className="bg-white shadow rounded p-4">
          <h2 className="text-sm font-semibold text-gray-600">Average Risk</h2>
          <p className="text-2xl font-bold text-blue-600">{average.toFixed(2)}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded p-4 max-h-[300px]">
          <h3 className="text-md font-semibold mb-2">Risk Scores by Asset</h3>
          <div className="relative h-[250px]">
            <Bar data={barData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>
        <div className="bg-white shadow rounded p-4 max-h-[300px]">
          <h3 className="text-md font-semibold mb-2">Category Distribution</h3>
          <div className="relative h-[250px]">
            <Pie data={pieData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>
      </div>

      {/* Risk Time Series Chart */}
      {selectedSymbol && riskTimeSeries.length > 0 && (
        <div className="bg-white shadow rounded p-4">
          <h3 className="text-md font-semibold mb-2">{selectedSymbol} Risk Over Time</h3>
          <div className="relative h-[300px]">
            <Line data={lineData} options={lineOptions} />
          </div>
        </div>
      )}

      {/* Table */}
      <div className="overflow-auto rounded border shadow max-h-[400px]">
        <table className="min-w-full bg-white text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left">Symbol</th>
              <th className="px-4 py-3 text-left">Score</th>
              <th className="px-4 py-3 text-left">Category</th>
              <th className="px-4 py-3 text-left">Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((p, idx) => (
              <tr
                key={idx}
                onClick={() => setSelectedSymbol(p.symbol)}
                className="border-t hover:bg-gray-100 cursor-pointer"
              >
                <td className="px-4 py-2 font-semibold">{p.symbol}</td>
                <td className="px-4 py-2">{p.risk_score.toFixed(2)}</td>
                <td className="px-4 py-2">
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded-full ${riskColors[p.risk_category as keyof typeof riskColors]}`}
                  >
                    {p.risk_category}
                  </span>
                </td>
                <td className="px-4 py-2 text-gray-500">
                  {new Date(p.analysis_timestamp).toLocaleString("en-US", {
                    year: "numeric",
                    month: "short",
                    day: "numeric",
                    hour: "numeric",
                    minute: "2-digit"
                  })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
