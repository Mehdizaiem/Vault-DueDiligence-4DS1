"use client";

import { Upload, FileText, Search, Filter } from "lucide-react";

export default function DocumentsPage() {
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div className="space-y-4">
          <h2 className="text-3xl font-bold tracking-tight">Document Analysis</h2>
          <p className="text-muted-foreground">
            Automated extraction and analysis of crypto fund documentation
          </p>
        </div>
        <div className="flex gap-4">
          <button className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800">
            <Upload size={20} />
            Upload Documents
          </button>
        </div>
      </div>

      {/* Search and Filter Section */}
      <div className="mt-8 flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-3 text-gray-500" size={20} />
          <input 
            type="text"
            placeholder="Search documents..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg"
          />
        </div>
        <button className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50">
          <Filter size={20} />
          Filters
        </button>
      </div>

      {/* Documents Grid */}
      <div className="mt-8 grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
        {Array.from({length: 6}).map((_, i) => (
          <div key={i} className="bg-white p-6 rounded-xl shadow-sm border hover:shadow-md transition cursor-pointer">
            <div className="flex items-center gap-3 mb-4">
              <FileText className="h-8 w-8 text-blue-500" />
              <div>
                <h3 className="font-semibold">Document {i + 1}</h3>
                <p className="text-sm text-gray-500">PDF • 2.4 MB</p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-sm">
                <span className="font-medium">Analysis Status:</span>
                <span className="ml-2 text-green-600">Completed</span>
              </div>
              <div className="text-sm">
                <span className="font-medium">Last Updated:</span>
                <span className="ml-2 text-gray-500">2 hours ago</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}