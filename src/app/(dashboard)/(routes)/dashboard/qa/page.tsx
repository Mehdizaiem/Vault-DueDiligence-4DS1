"use client";

import { MessageSquare, Send } from "lucide-react";

export default function QAPage() {
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <h2 className="text-3xl font-bold tracking-tight">Q&A System</h2>
        <p className="text-muted-foreground">
          Get instant answers to due diligence queries
        </p>
      </div>

      {/* Recent Questions */}
      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-4">Recent Questions</h3>
        <div className="space-y-4">
          {Array.from({length: 3}).map((_, i) => (
            <div key={i} className="bg-white p-6 rounded-xl shadow-sm border">
              <div className="flex items-start gap-4">
                <MessageSquare className="h-6 w-6 text-blue-500 mt-1" />
                <div className="flex-1">
                  <h4 className="font-medium">Sample Question {i + 1}</h4>
                  <p className="text-sm text-gray-500 mt-1">
                    What are the risk management procedures for Fund XYZ?
                  </p>
                  <div className="mt-4 bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm">
                      Based on the available documentation, Fund XYZ implements...
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Question Input */}
      <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 w-[80%] max-w-4xl">
        <div className="bg-white rounded-xl shadow-lg border p-4">
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="Ask a question..."
              className="flex-1 border rounded-lg px-4 py-2"
            />
            <button className="bg-black text-white px-6 py-2 rounded-lg hover:bg-gray-800 flex items-center gap-2">
              <Send size={20} />
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}