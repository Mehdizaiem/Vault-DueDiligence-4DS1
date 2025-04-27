// File: components/RiskDetailsModel.tsx
"use client";

import React from "react";

interface RiskDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  riskScore: number;
  riskCategory: string;
  riskFactors: string[];
  title: string;
}

export const RiskDetailsModal: React.FC<RiskDetailsModalProps> = ({
  isOpen,
  onClose,
  riskScore,
  riskCategory,
  riskFactors,
  title,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-lg w-full">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold truncate max-w-[80%]" title={title}>
            {title} - Risk Analysis
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-red-500 text-xl font-bold"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        <div className="space-y-4 p-4">
        <div className="flex items-center justify-center mt-4">
        <div className="relative">
          <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" stroke="gray" strokeWidth="10" fill="none" />
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="blue"
              strokeWidth="10"
              strokeDasharray="283"
              strokeDashoffset={283 - (riskScore / 100) * 283}
              fill="none"
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-lg font-semibold">{riskScore.toFixed(0)}%</span>
          </div>
        </div>
      </div>


          <div className="text-sm">
            <strong>Risk Category:</strong>{" "}
            <span className="text-blue-600">{riskCategory || "N/A"}</span>
          </div>

          {riskFactors?.length > 0 ? (
            <div className="mt-2">
              <strong>Risk Factors:</strong>
              <ul className="list-disc list-inside text-xs text-gray-600 mt-1">
                {riskFactors.map((factor, i) => (
                  <li key={i}>{factor}</li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="text-xs text-gray-500 mt-2">
              No specific risk factors found.
            </div>
          )}
        </div>

        <div className="mt-6 text-right">
          <button
            onClick={onClose}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
