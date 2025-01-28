"use client";

import { AlertTriangle} from "lucide-react";

export default function RiskPage() {
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <h2 className="text-3xl font-bold tracking-tight">Risk Analysis</h2>
        <p className="text-muted-foreground">
          Monitor real-time risk metrics and alerts
        </p>
      </div>

      {/* Risk Metrics */}
      <div className="mt-8 grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Overall Risk Score", value: "94%", trend: "+2.1%", color: "text-green-500" },
          { label: "Compliance Score", value: "88%", trend: "-0.8%", color: "text-red-500" },
          { label: "Active Alerts", value: "7", trend: "+3", color: "text-orange-500" },
          { label: "Monitored Funds", value: "28", trend: "+5", color: "text-blue-500" },
        ].map((metric, i) => (
          <div key={i} className="bg-white p-6 rounded-xl shadow-sm border">
            <h3 className="text-sm text-gray-500">{metric.label}</h3>
            <div className="flex items-center justify-between mt-2">
              <span className="text-2xl font-bold">{metric.value}</span>
              <span className={`text-sm ${metric.color}`}>{metric.trend}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Active Alerts */}
      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-4">Active Alerts</h3>
        <div className="space-y-4">
          {Array.from({length: 3}).map((_, i) => (
            <div key={i} className="bg-white p-6 rounded-xl shadow-sm border">
              <div className="flex items-start gap-4">
                <AlertTriangle className="h-6 w-6 text-orange-500 mt-1" />
                <div>
                  <h4 className="font-medium">Risk Alert {i + 1}</h4>
                  <p className="text-sm text-gray-500 mt-1">
                    Unusual transaction pattern detected in Fund XYZ
                  </p>
                  <div className="flex gap-2 mt-2">
                    <span className="text-xs bg-orange-100 text-orange-600 px-2 py-1 rounded-full">
                      High Priority
                    </span>
                    <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                      2 hours ago
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}