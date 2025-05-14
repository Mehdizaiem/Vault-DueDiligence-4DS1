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
  const [selectedRange, setSelectedRange] = useState("30d"); // default selected
  const [error, setError] = useState("");
  const [filterText, setFilterText] = useState("");
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({ start: "", end: "" });
  const [filterOpen, setFilterOpen] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState(false);
  type RiskProfileWithAlerts = RiskProfile & { alerts?: string[] };
  const [alertsOpen, setAlertsOpen] = useState(false);



useEffect(() => {
  fetch("/api/risk-profiles")
    .then((res) => res.json())
    .then((data) => {
        if (data.success) setAllProfiles(data.data);
        else setError("Failed to load risk data.");
    })
    .catch((err) => {
      console.error("Fetch error:", err);
      setError("Error connecting to API.");
    });
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

    const alertMap: { [key: string]: boolean } = {};
    const uniqueAlerts: string[] = [];

    filtered.forEach((p) => {
      const alerts: string[] = [];

      alerts.push(`ℹ️ ${p.symbol} score: ${p.risk_score.toFixed(2)} (${p.risk_category})`);

      if (p.risk_score >= 80) {
        alerts.push(`🚨 High Risk Detected for ${p.symbol} — Score: ${p.risk_score.toFixed(1)} (${p.risk_category})`);
      }

      if (p.risk_factors?.some((f) => f.toLowerCase().includes("sentiment"))) {
        alerts.push(`⚠️ Sentiment-related risk for ${p.symbol}`);
      }

      alerts.forEach((alert) => {
        if (!alertMap[alert]) {
          alertMap[alert] = true;
          uniqueAlerts.push(alert);
        }
      });
    });

    // Deduplicate alerts
    //const uniqueAlerts = [...new Set(filteredWithAlerts.flatMap((p) => p.alerts ?? []))];





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
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Risk Analysis Dashboard</h1>
      
      {error && <p className="text-red-600">{error}</p>}

      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
      <div className="flex items-center border rounded overflow-hidden text-sm">
      
      {/* Custom Date Range */}
      <div className="flex items-center gap-2">
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
      </div>
      
      {/* Filter & Export */}
      <div className="flex items-center justify-end gap-4">
      {/* Alert Button */}
      <div className="relative">
        <button
          onClick={() => setAlertsOpen(!alertsOpen)}
          className="relative text-gray-700 hover:text-yellow-600 focus:outline-none"
        >
          🔔
          {uniqueAlerts.length > 0 && (
            <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">
              {uniqueAlerts.length}
            </span>
          )}
        </button>

        {alertsOpen && (
          <div className="absolute right-0 mt-2 w-80 bg-white border rounded shadow-lg z-50 max-h-96 overflow-auto">
            <div className="p-3 border-b font-semibold text-gray-800">Alerts</div>
            <ul className="divide-y divide-gray-100">
              {uniqueAlerts.length > 0 ? (
                uniqueAlerts.map((alert, i) => (
                  <li key={i} className="p-3 text-sm text-gray-700 hover:bg-gray-50">
                    {alert}
                  </li>
                ))
              ) : (
                <li className="p-3 text-sm text-gray-500 italic">No alerts</li>
              )}
            </ul>
          </div>
        )}
      </div>

      {/* Filter */}
      <div className="relative">
        <button
          onClick={() => setFilterOpen(!filterOpen)}
          className="border px-4 py-2 rounded text-sm flex items-center gap-2 hover:bg-gray-50"
        >
          <span>Filter</span>
          <span className="ml-1">▾</span>
        </button>
        {filterOpen && (
          <div className="absolute z-10 mt-2 bg-white border rounded shadow p-3 w-64">
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter by symbol or category"
              className="w-full border p-2 rounded text-sm"
            />
          </div>
        )}
      </div>

      {/* Export */}
      <button
        onClick={exportCSV}
        className="border px-4 py-2 rounded text-sm hover:bg-gray-50 flex items-center gap-2"
      >
        ⬇ Export
      </button>
    </div>
</div>


<div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
  {/* Highest Risk Card */}
  <div className="relative overflow-hidden rounded-xl border bg-white/50 backdrop-blur-lg p-6 transition-all duration-300 hover:shadow-xl hover:scale-[1.02] shadow-lg group-hover:shadow-blue-500/25">
    <div className="absolute -left-12 -bottom-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-0 from-red-500 to-yellow-600" />
    <div>
      <div className="text-sm font-medium text-gray-500 mb-1">Highest Risk</div>
      <div className="text-2xl font-bold text-red-600">
        {highest?.symbol} ({highest?.risk_score?.toFixed(2)})
      </div>
      <span
        className={`text-xs font-medium mt-1 inline-block px-2 py-1 rounded ${riskColors[highest?.risk_category as keyof typeof riskColors]}`}
      >
        {highest?.risk_category}
      </span>
    </div>
    <div className="absolute -right-12 -top-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-20 from-blue-500 to-indigo-600" />
  </div>

  {/* Lowest Risk Card */}
 <div className="relative overflow-hidden rounded-xl border bg-white/50 backdrop-blur-lg p-6 transition-all duration-300 hover:shadow-xl hover:scale-[1.02] shadow-lg group-hover:shadow-blue-500/25">
    <div className="absolute -left-12 -bottom-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-0 from-red-500 to-yellow-600" />
    <div>
      <div className="text-sm font-medium text-gray-500 mb-1">Lowest Risk</div>
      <div className="text-2xl font-bold text-green-600">
        {lowest?.symbol} ({lowest?.risk_score?.toFixed(2)})
      </div>
      <span
        className={`text-xs font-medium mt-1 inline-block px-2 py-1 rounded ${riskColors[lowest?.risk_category as keyof typeof riskColors]}`}
      >
        {lowest?.risk_category}
      </span>
    </div>
        <div className="absolute -right-12 -top-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-20 from-blue-500 to-indigo-600" />

  </div>

  {/* Average Risk Card */}
<div className="relative overflow-hidden rounded-xl border bg-white/50 backdrop-blur-lg p-6 transition-all duration-300 hover:shadow-xl hover:scale-[1.02] shadow-lg group-hover:shadow-blue-500/25">
    <div className="absolute -left-12 -bottom-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-0 from-red-500 to-yellow-600" />
        <div>
      <div className="text-sm font-medium text-gray-500 mb-1">Average Risk</div>
      <div className="text-3xl font-bold text-blue-600">{average.toFixed(2)}</div>
    </div>
            <div className="absolute -right-12 -top-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-20 from-blue-500 to-indigo-600" />

  </div>
</div>


      {/* Charts */}
      {/*<div className="rounded-lg border text-card-foreground overflow-hidden bg-white/50 backdrop-blur-lg shadow-xl border-none hover:shadow-2xl transition-all duration-300"></div>*/}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 rounded-lg text-card-foreground bg-white/50 backdrop-blur-lg shadow-xl p-4 transition-all duration-300">
      {/* Risk Scores Bar Chart */}
      <div className="bg-white shadow rounded p-4 h-[300px]">
        <h3 className="text-md font-semibold mb-2">Risk Scores by Asset</h3>
        <div className="relative h-[250px]">
          <Bar data={barData} options={{ maintainAspectRatio: false }} />
        </div>
      </div>

      {/* Category Distribution Pie Chart */}
      <div className="bg-white shadow rounded p-4 h-[300px]">
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
      <div className="rounded-2xl border border-gray-200 shadow-md overflow-hidden">
      <table className="min-w-full divide-y divide-gray-200 text-sm text-gray-700 bg-white">
        <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-500">
          <tr>
            <th className="px-6 py-3 text-left">Symbol</th>
            <th className="px-6 py-3 text-left">Score</th>
            <th className="px-6 py-3 text-left">Category</th>
            <th className="px-6 py-3 text-left">Timestamp</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {filtered.map((p, idx) => (
            <tr
              key={idx}
              onClick={() => setSelectedSymbol(p.symbol)}
              className="hover:bg-gray-50 transition-colors duration-150 cursor-pointer"
            >
              <td className="px-6 py-3 font-medium text-gray-800">{p.symbol}</td>
              <td className="px-6 py-3">{p.risk_score.toFixed(2)}</td>
              <td className="px-6 py-3">
                <span className={`px-2 py-1 rounded-full text-xs font-semibold ${riskColors[p.risk_category as keyof typeof riskColors]}`}>
                  {p.risk_category}
                </span>
              </td>
              <td className="px-6 py-3 text-gray-500">
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
