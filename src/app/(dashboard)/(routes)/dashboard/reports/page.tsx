"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  FileText,
  Download,
  Trash2,
  Plus,
  Search,
  CheckCircle,
  Clock,
  AlertCircle,
  Calendar,
  BarChart,
  FileBarChart,
  Users,
  Briefcase,
  RefreshCw,
  Zap,
  Settings,
  CircleDollarSign,
  Star,
  Shield,
  Gift,
  BookOpen,
  FileOutput,
  Lock
} from 'lucide-react';

// Import UI components
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from 'framer-motion';
import CustomizeTemplateModal from '@/components/reports/CustomizeTemplateModal';
import TemplatesList from '@/components/reports/TemplatesList';
import { getTemplates, saveTemplate, deleteTemplate } from '@/services/templates-service';
import { downloadReportAsPPTX } from '@/utils/report-downloader';

// Sample templates
const REPORT_TEMPLATES = [
  {
    id: 'comprehensive',
    title: 'Comprehensive Due Diligence',
    description: 'Full analysis including risk assessment, compliance check, and investment analysis',
    icon: FileBarChart,
    sections: 12,
    avgPages: '40-60',
    duration: '15-20 min',
    color: 'bg-blue-500'
  },
  {
    id: 'compliance',
    title: 'Regulatory Compliance',
    description: 'Deep dive into regulatory requirements and compliance status',
    icon: Shield,
    sections: 8,
    avgPages: '25-35',
    duration: '10-15 min',
    color: 'bg-green-500'
  },
  {
    id: 'risk',
    title: 'Risk Assessment',
    description: 'Focused analysis on potential risks and mitigation strategies',
    icon: AlertCircle,
    sections: 6,
    avgPages: '15-25',
    duration: '8-12 min',
    color: 'bg-orange-500'
  },
  {
    id: 'comparative',
    title: 'Comparative Analysis',
    description: 'Side-by-side comparison of multiple funds or assets',
    icon: BarChart,
    sections: 9,
    avgPages: '30-50',
    duration: '12-18 min',
    color: 'bg-purple-500'
  },
  {
    id: 'security',
    title: 'Security Audit',
    description: 'Detailed examination of security measures and vulnerabilities',
    icon: Lock,
    sections: 7,
    avgPages: '20-30',
    duration: '10-14 min',
    color: 'bg-red-500'
  }
];

const ReportStatusBadge = ({ status }: { status: string }) => {
  let color;
  let icon;
  
  switch (status) {
    case 'completed':
      color = 'bg-green-100 text-green-800';
      icon = <CheckCircle className="h-3.5 w-3.5 text-green-600" />;
      break;
    case 'processing':
      color = 'bg-blue-100 text-blue-800';
      icon = <Clock className="h-3.5 w-3.5 text-blue-600 animate-pulse" />;
      break;
    case 'pending':
      color = 'bg-yellow-100 text-yellow-800';
      icon = <Clock className="h-3.5 w-3.5 text-yellow-600" />;
      break;
    case 'failed':
      color = 'bg-red-100 text-red-800';
      icon = <AlertCircle className="h-3.5 w-3.5 text-red-600" />;
      break;
    default:
      color = 'bg-gray-100 text-gray-800';
      icon = <Clock className="h-3.5 w-3.5 text-gray-600" />;
  }
  
  return (
    <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${color}`}>
      {icon}
      <span className="capitalize">{status}</span>
    </span>
  );
};

const formatDate = (dateString: string) => {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
};

export default function ReportsPage() {
  const [reports, setReports] = useState([
    {
      id: '1',
      title: 'Q1 2025 Bitcoin Trust Fund Analysis',
      status: 'completed',
      type: 'comprehensive',
      created_at: '2025-03-15T10:30:00Z',
      updated_at: '2025-03-15T10:45:00Z',
      document_count: 8,
      page_count: 42,
      author: 'System',
      risk_score: 65,
      compliance_score: 78,
      entities: ['Bitcoin', 'Tether', 'Ethereum']
    },
    {
      id: '2',
      title: 'Emerging Crypto Funds Comparative Analysis',
      status: 'completed',
      type: 'comparative',
      created_at: '2025-04-22T09:15:00Z',
      updated_at: '2025-04-22T09:35:00Z',
      document_count: 15,
      page_count: 78,
      author: 'System',
      risk_score:  34,
      compliance_score:  96,
      entities: ['Solana', 'Avalanche', 'Polygon', 'Cardano']
    },
    {
      id: '3',
      title: 'Regulatory Compliance Assessment - USDC Treasury',
      status: 'processing',
      type: 'compliance',
      created_at: '2025-05-12T14:20:00Z',
      updated_at: '2025-05-13T14:45:00Z',
      document_count: 12,
      page_count: 56,
      author: 'System',
      risk_score: 0,
      compliance_score: 0,
      entities: ['USDC', 'Circle', 'Treasury']
    },
    {
      id: '4',
      title: 'DeFi Exposure Analysis - Institutional Portfolio',
      status: 'pending',
      type: 'risk',
      created_at: '2025-05-13T16:10:00Z',
      updated_at: '2025-05-14T16:25:00Z',
      document_count: 5,
      page_count: 0,
      author: 'System',
      risk_score:  0,
      compliance_score:  0,
      entities: ['Uniswap', 'Aave', 'Compound']
    },
    {
      id: '5',
      title: 'Cross-Chain Integration Security Analysis',
      status: 'pending',
      type: 'security',
      created_at: '2025-05-14T08:30:00Z',
      updated_at: '2025-05-14T12:30:00Z',
      document_count: 9,
      page_count: 0,
      author: 'System',
      risk_score: 0,
      compliance_score: 0,
      entities: ['Wormhole', 'Axelar', 'LayerZero']
    }
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [isCreatingReport, setIsCreatingReport] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [isCustomizeTemplateModalOpen, setIsCustomizeTemplateModalOpen] = useState(false);
  const [templateToEdit, setTemplateToEdit] = useState<any>(null);
  const [templates, setTemplates] = useState<any[]>([]);
  const [downloadingReportId, setDownloadingReportId] = useState<string | null>(null);

  // Sample document options for selection
  const availableDocuments = [
    { id: 'doc1', title: 'Bitcoin Trust Fund Whitepaper', size: '2.4 MB', type: 'PDF' },
    { id: 'doc2', title: 'Quarterly Financial Statement Q1 2024', size: '3.1 MB', type: 'DOCX' },
    { id: 'doc3', title: 'Risk Management Framework', size: '1.8 MB', type: 'PDF' },
    { id: 'doc4', title: 'Compliance Certification', size: '0.9 MB', type: 'PDF' },
    { id: 'doc5', title: 'Investment Strategy Overview', size: '2.2 MB', type: 'DOCX' },
    { id: 'doc6', title: 'Custody Agreement', size: '1.5 MB', type: 'PDF' },
    { id: 'doc7', title: 'Key Management Protocol', size: '1.1 MB', type: 'PDF' },
    { id: 'doc8', title: 'Regulatory Filing 2024', size: '4.2 MB', type: 'PDF' },
  ];
  
  // Fetch templates
  useEffect(() => {
    async function fetchTemplates() {
      try {
        const templateData = await getTemplates();
        setTemplates(templateData);
      } catch (error) {
        console.error('Error fetching templates:', error);
      }
    }
    
    fetchTemplates();
  }, []);
  
  // Filter reports based on search query and filters
  const filteredReports = reports.filter(report => {
    const matchesSearch = report.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                         report.entities.some(entity => entity.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesStatus = statusFilter === 'all' || report.status === statusFilter;
    const matchesType = typeFilter === 'all' || report.type === typeFilter;
    
    return matchesSearch && matchesStatus && matchesType;
  });
  
  // Toggle document selection
  const toggleDocument = (docId: string) => {
    setSelectedDocuments(prev => 
      prev.includes(docId) 
        ? prev.filter(id => id !== docId) 
        : [...prev, docId]
    );
  };
  
  // Generate a new report
  const handleGenerateReport = () => {
    if (!selectedTemplate || selectedDocuments.length === 0) {
      alert('Please select a template and at least one document.');
      return;
    }
    
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      const newReport = {
        id: `${reports.length + 1}`,
        title: `New ${selectedTemplate.charAt(0).toUpperCase() + selectedTemplate.slice(1)} Report`,
        status: 'processing',
        type: selectedTemplate,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        document_count: selectedDocuments.length,
        page_count: 0,
        author: 'System',
        risk_score: 0,
        compliance_score: 0,
        entities: ['Bitcoin', 'Ethereum'] // Placeholder
      };
      
      setReports([newReport, ...reports]);
      setIsCreatingReport(false);
      setSelectedTemplate(null);
      setSelectedDocuments([]);
      setLoading(false);
    }, 2000);
  };
  
  // Handle template edit
  const handleEditTemplate = (template: any) => {
    // Set the template data in the form
    setTemplateToEdit(template);
    // Open the modal
    setIsCustomizeTemplateModalOpen(true);
  };
  
  // Handle template save
  const handleSaveTemplate = async (template: any) => {
    try {
      // If editing an existing template, update it, otherwise create a new one
      if (templateToEdit && templateToEdit.id) {
        // For demo purposes, we'll just delete the old one and create a new one
        await deleteTemplate(templateToEdit.id);
      }
      
      // Save the template
      await saveTemplate(template);
      
      // Refetch templates
      const updatedTemplates = await getTemplates();
      setTemplates(updatedTemplates);
      
      // Close the modal and reset the template to edit
      setIsCustomizeTemplateModalOpen(false);
      setTemplateToEdit(null);
      
      // Show success message
      alert(`Template "${template.name}" saved successfully!`);
    } catch (error) {
      console.error('Error saving template:', error);
      alert('Failed to save template. Please try again.');
    }
  };
  
  // Handle template modal close
  const handleCloseTemplateModal = () => {
    setIsCustomizeTemplateModalOpen(false);
    setTemplateToEdit(null);
  };
  // 3. Add the handleDownloadReport function inside the ReportsPage component
  const handleDownloadReport = async (report: any) => {
    // Set the downloading state to show loading
    setDownloadingReportId(report.id);
    
    try {
      // Call the utility function to download the report
      const success = await downloadReportAsPPTX({
        title: report.title,
        type: report.type,
        reportId: report.id,
        report: report // Pass the full report object
      });
      
      if (!success) {
        alert('Failed to download report. Please try again.');
      }
    } catch (error) {
      console.error('Error downloading report:', error);
      alert('An error occurred while downloading the report.');
    } finally {
      // Reset the downloading state
      setDownloadingReportId(null);
    }
  };
  return (
    <div className="flex-1 p-8 pt-6 bg-gradient-to-br from-gray-50 via-blue-50/10 to-purple-50/10">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Report Generation
            </h1>
            <p className="text-muted-foreground">
              Generate comprehensive due diligence reports from analyzed documents
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsCreatingReport(true)}
              className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition-colors"
            >
              <Plus size={16} />
              <span>New Report</span>
            </button>
            <button
              onClick={() => {
                setTemplateToEdit(null);
                setIsCustomizeTemplateModalOpen(true);
              }}
              className="flex items-center gap-2 border bg-white px-4 py-2 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <Settings size={16} />
              <span>Customize Templates</span>
            </button>
            <Link
              href="/dashboard/reports/history"
              className="flex items-center gap-2 border bg-white px-4 py-2 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <FileText size={16} />
              <span>View History</span>
            </Link>
          </div>
        </div>
        
        {/* Generate New Report Panel */}
        {isCreatingReport && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="border bg-white/90 backdrop-blur-sm shadow-lg overflow-hidden">
              <CardHeader className="bg-gray-50/80 border-b pb-8">
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-xl font-bold">Create New Report</CardTitle>
                    <CardDescription>Generate a comprehensive analysis based on your documents</CardDescription>
                  </div>
                  <button 
                    onClick={() => setIsCreatingReport(false)}
                    className="text-gray-500 hover:text-gray-800 transition-colors"
                  >
                    &times;
                  </button>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-6">
                  {/* Template Selection */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">1. Select Report Template</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {REPORT_TEMPLATES.map(template => (
                        <div
                          key={template.id}
                          onClick={() => setSelectedTemplate(template.id)}
                          className={`border rounded-xl p-4 cursor-pointer transition-all duration-200 ${
                            selectedTemplate === template.id 
                              ? 'border-blue-500 bg-blue-50/50 shadow-md scale-[1.02]' 
                              : 'hover:border-gray-300 hover:bg-gray-50'
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            <div className={`p-2 rounded-lg ${template.color} bg-opacity-10`}>
                              <template.icon className={`h-5 w-5 ${template.color.replace('bg-', 'text-')}`} />
                            </div>
                            <div>
                              <h4 className="font-medium text-gray-900">{template.title}</h4>
                              <p className="text-sm text-gray-500 mt-1">{template.description}</p>
                              <div className="flex flex-wrap gap-2 mt-2">
                                <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">
                                  {template.sections} sections
                                </span>
                                <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">
                                  {template.avgPages} pages
                                </span>
                                <span className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full">
                                  {template.duration}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Document Selection */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">2. Select Documents to Analyze</h3>
                    <div className="border rounded-lg overflow-hidden">
                      <div className="bg-gray-50 p-3 border-b flex justify-between items-center">
                        <div className="text-sm font-medium text-gray-700">
                          {selectedDocuments.length} documents selected
                        </div>
                        <div className="flex items-center">
                          <input 
                            type="text" 
                            placeholder="Search documents..." 
                            className="text-sm border rounded-lg px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                        </div>
                      </div>
                      <div className="divide-y max-h-[300px] overflow-y-auto">
                        {availableDocuments.map(doc => (
                          <div 
                            key={doc.id}
                            className={`flex items-center gap-3 p-3 hover:bg-gray-50 transition-colors ${
                              selectedDocuments.includes(doc.id) ? 'bg-blue-50' : ''
                            }`}
                          >
                            <input 
                              type="checkbox"
                              checked={selectedDocuments.includes(doc.id)}
                              onChange={() => toggleDocument(doc.id)}
                              className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                            />
                            <div className="flex-1">
                              <h4 className="font-medium text-gray-900 text-sm">{doc.title}</h4>
                              <div className="flex items-center gap-2 mt-1">
                                <span className="text-xs text-gray-500">{doc.type}</span>
                                <span className="text-xs text-gray-400">•</span>
                                <span className="text-xs text-gray-500">{doc.size}</span>
                              </div>
                            </div>
                            <div>
                              <FileText className="h-5 w-5 text-gray-400" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="bg-gray-50/80 px-6 py-4 border-t">
                <div className="flex justify-between items-center w-full">
                  <button
                    onClick={() => setIsCreatingReport(false)}
                    className="text-gray-700 hover:text-gray-900"
                  >
                    Cancel
                  </button>
                  
                  <button
                    onClick={handleGenerateReport}
                    disabled={!selectedTemplate || selectedDocuments.length === 0 || loading}
                    className={`flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors ${
                      (!selectedTemplate || selectedDocuments.length === 0 || loading) ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        <span>Generating...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4" />
                        <span>Generate Report</span>
                      </>
                    )}
                  </button>
                </div>
              </CardFooter>
            </Card>
          </motion.div>
        )}
        
        {/* Latest Reports Section */}
        <section>
          <h2 className="text-xl font-semibold mb-4">Latest Reports</h2>
          
          {/* Filters */}
          <div className="bg-white border rounded-lg p-4 mb-6 flex flex-wrap gap-4 items-center justify-between">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
              <input
                type="text"
                placeholder="Search reports by title or cryptocurrency..."
                className="w-full border rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <div className="flex items-center gap-2 text-sm">
              <div>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Statuses</option>
                  <option value="completed">Completed</option>
                  <option value="processing">Processing</option>
                  <option value="pending">Pending</option>
                  <option value="failed">Failed</option>
                </select>
              </div>
              
              <div>
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                  className="border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Types</option>
                  <option value="comprehensive">Comprehensive</option>
                  <option value="compliance">Compliance</option>
                  <option value="risk">Risk</option>
                  <option value="comparative">Comparative</option>
                  <option value="security">Security</option>
                </select>
              </div>
            </div>
          </div>
          
          {/* Reports Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredReports.map(report => (
              <Link href={`/dashboard/reports/${report.id}`} key={report.id}>
                <Card className="bg-white/90 backdrop-blur-sm hover:shadow-lg transition-shadow duration-200 border overflow-hidden h-full flex flex-col">
                  <CardHeader className="border-b pb-3">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-lg font-bold line-clamp-1" title={report.title}>
                        {report.title}
                      </CardTitle>
                      <ReportStatusBadge status={report.status} />
                    </div>
                    <CardDescription className="flex items-center gap-2 mt-2">
                      <Calendar className="h-3.5 w-3.5 text-gray-400" />
                      <span>{formatDate(report.created_at)}</span>
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="flex-1 pt-4">
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <div className="text-sm text-gray-500 mb-1">Risk Score</div>
                        <div className={`text-xl font-bold ${report.risk_score > 70 ? 'text-red-600' : report.risk_score > 40 ? 'text-orange-500' : 'text-green-600'}`}>
                          {report.risk_score || 'N/A'}
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <div className="text-sm text-gray-500 mb-1">Compliance</div>
                        <div className={`text-xl font-bold ${report.compliance_score < 60 ? 'text-red-600' : report.compliance_score < 80 ? 'text-orange-500' : 'text-green-600'}`}>
                          {report.compliance_score || 'N/A'}
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">Type:</span>
                        <span className="capitalize font-medium">{report.type}</span>
                      </div>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">Documents:</span>
                        <span className="font-medium">{report.document_count}</span>
                      </div>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">Pages:</span>
                        <span className="font-medium">{report.page_count || '-'}</span>
                      </div>
                    </div>
                    
                    {/* Entities */}
                    <div className="mt-4">
                      <div className="text-sm text-gray-500 mb-2">Entities</div>
                      <div className="flex flex-wrap gap-2">
                        {report.entities.map(entity => (
                          <Badge key={entity} className="bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200">
                            {entity}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                  
                  <CardFooter className="border-t p-4">
                    <div className="w-full flex justify-between items-center">
                      <div className="text-sm text-gray-500">
                        Last Updated: {formatDate(report.updated_at)}
                      </div>
                      <div className="flex gap-2">
                        {report.status === 'completed' && (
                          <button 
                            className="flex items-center gap-1.5 text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
                            onClick={(e) => {
                              e.preventDefault(); // Prevent navigation
                              handleDownloadReport(report);
                            }}
                            disabled={downloadingReportId === report.id}
                          >
                            {downloadingReportId === report.id ? (
                              <>
                                <RefreshCw size={14} className="animate-spin" />
                                <span>Downloading...</span>
                              </>
                            ) : (
                              <>
                                <Download size={14} />
                                <span>Download</span>
                              </>
                            )}
                          </button>
                        )}
                        <button className="flex items-center gap-1.5 text-sm font-medium text-red-600 hover:text-red-800 transition-colors">
                          <Trash2 size={14} />
                          <span>Delete</span>
                        </button>
                      </div>
                    </div>
                  </CardFooter>
                </Card>
              </Link>
            ))}
          </div>
          
          {/* Empty State */}
          {filteredReports.length === 0 && (
            <div className="bg-white border border-dashed rounded-lg p-8 text-center">
              <FileOutput className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-700 mb-2">No reports found</h3>
              <p className="text-gray-500 mb-6">
                {searchQuery || statusFilter !== 'all' || typeFilter !== 'all'
                  ? 'Try adjusting your filters or creating a new report'
                  : 'Generate your first due diligence report to get started'}
              </p>
              <button
                onClick={() => setIsCreatingReport(true)}
                className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus size={16} />
                <span>New Report</span>
              </button>
            </div>
          )}
        </section>
        
        {/* Custom Templates Section */}
        <section className="mt-12">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold">Custom Templates</h2>
            <button
              onClick={() => {setTemplateToEdit(null);
                setIsCustomizeTemplateModalOpen(true);
              }}
              className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition-colors"
            >
              <Plus size={16} />
              <span>New Template</span>
            </button>
          </div>
          
          <TemplatesList 
            templates={templates}
            onRefresh={async () => {
              const updatedTemplates = await getTemplates();
              setTemplates(updatedTemplates);
            }}
            onEdit={handleEditTemplate}
          />
        </section>
        
        {/* Analytics Summary */}
        <section className="mt-12">
          <h2 className="text-xl font-semibold mb-4">Report Analytics</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-blue-50 border-none overflow-hidden">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium text-blue-800">Risk Overview</h3>
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <AlertCircle className="h-5 w-5 text-blue-600" />
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-green-700">Fully Compliant</span>
                    <span className="font-bold text-green-900">18</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-green-700">Partial Compliance</span>
                    <span className="font-bold text-green-900">5</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-green-700">Non-Compliant</span>
                    <span className="font-bold text-green-900">2</span>
                  </div>
                  
                  <div className="w-full h-2 bg-green-200 rounded-full overflow-hidden mt-6">
                    <div 
                      className="h-full bg-green-600 rounded-full"
                      style={{ width: '82%' }}
                    />
                  </div>
                  <div className="text-sm text-green-700 text-center">
                    82% overall compliance rate
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-purple-50 border-none overflow-hidden">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium text-purple-800">Report Activity</h3>
                  <div className="bg-purple-100 p-2 rounded-lg">
                    <FileBarChart className="h-5 w-5 text-purple-600" />
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-purple-700">Reports Generated</span>
                    <span className="font-bold text-purple-900">42</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-purple-700">Documents Analyzed</span>
                    <span className="font-bold text-purple-900">267</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-purple-700">Pages Processed</span>
                    <span className="font-bold text-purple-900">1,845</span>
                  </div>
                  
                  <div className="w-full h-2 bg-purple-200 rounded-full overflow-hidden mt-6">
                    <div 
                      className="h-full bg-purple-600 rounded-full"
                      style={{ width: '78%' }}
                    />
                  </div>
                  <div className="text-sm text-purple-700 text-center">
                    78% increase in report generation
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-orange-50 border-none overflow-hidden">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium text-orange-800">Template Usage</h3>
                  <div className="bg-orange-100 p-2 rounded-lg">
                    <Settings className="h-5 w-5 text-orange-600" />
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-orange-700">Custom Templates</span>
                    <span className="font-bold text-orange-900">{templates.length}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-orange-700">Most Used Template</span>
                    <span className="font-bold text-orange-900">Compliance</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-orange-700">Template Efficiency</span>
                    <span className="font-bold text-orange-900">67%</span>
                  </div>
                  
                  <div className="w-full h-2 bg-orange-200 rounded-full overflow-hidden mt-6">
                    <div 
                      className="h-full bg-orange-600 rounded-full"
                      style={{ width: '67%' }}
                    />
                  </div>
                  <div className="text-sm text-orange-700 text-center">
                    67% of templates actively used
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
        
        {/* Integration Promotions */}
        <section className="mt-12">
          <h2 className="text-xl font-semibold mb-4">Enhance Your Reports</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-gradient-to-br from-black to-gray-800 text-white shadow-xl transition-transform duration-300 hover:scale-[1.01]">
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="bg-indigo-500 bg-opacity-30 p-3 rounded-xl">
                    <Briefcase className="h-8 w-8 text-indigo-400" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2">Enterprise Integration</h3>
                    <p className="text-gray-300 mb-4">
                      Connect with external data providers for enhanced due diligence reports with additional market insights and regulatory updates.
                    </p>
                    <button className="inline-flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors">
                      <Plus size={16} />
                      <span>Activate Integration</span>
                    </button>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-xl transition-transform duration-300 hover:scale-[1.01]">
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="bg-white bg-opacity-20 p-3 rounded-xl">
                    <Settings className="h-8 w-8 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2">Customize Report Templates</h3>
                    <p className="text-blue-100 mb-4">
                      Create your own report templates with custom branding, sections, and metrics tailored to your specific due diligence requirements.
                    </p>
                    <button 
                      onClick={() => {
                        setTemplateToEdit(null);
                        setIsCustomizeTemplateModalOpen(true);
                      }}
                      className="inline-flex items-center gap-2 bg-white text-blue-700 px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors"
                    >
                      <Settings size={16} />
                      <span>Customize Templates</span>
                    </button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
        
        {/* Premium Features Section */}
        <section className="mt-12">
          <h2 className="text-xl font-semibold mb-4">Premium Report Features</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="bg-gradient-to-br from-gray-50 to-white hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex flex-col items-center text-center">
                  <div className="bg-red-50 p-3 rounded-full mb-4">
                    <AlertCircle className="h-6 w-6 text-red-500" />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Advanced Risk Scoring</h3>
                  <p className="text-gray-600 text-sm">
                    Multi-factor risk assessment with detailed breakdown and mitigation recommendations.
                  </p>
                  <div className="mt-4">
                    <Badge className="bg-red-100 text-red-800 hover:bg-red-200">Premium</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-gray-50 to-white hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex flex-col items-center text-center">
                  <div className="bg-green-50 p-3 rounded-full mb-4">
                    <Users className="h-6 w-6 text-green-500" />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Multi-Team Collaboration</h3>
                  <p className="text-gray-600 text-sm">
                    Collaborative report editing with role-based permissions and approval workflows.
                  </p>
                  <div className="mt-4">
                    <Badge className="bg-green-100 text-green-800 hover:bg-green-200">Premium</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-gray-50 to-white hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex flex-col items-center text-center">
                  <div className="bg-blue-50 p-3 rounded-full mb-4">
                    <CircleDollarSign className="h-6 w-6 text-blue-500" />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Enhanced Financial Analysis</h3>
                  <p className="text-gray-600 text-sm">
                    Deep financial modeling with predictive analytics and historical comparisons.
                  </p>
                  <div className="mt-4">
                    <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-200">Premium</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-to-br from-gray-50 to-white hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex flex-col items-center text-center">
                  <div className="bg-purple-50 p-3 rounded-full mb-4">
                    <BookOpen className="h-6 w-6 text-purple-500" />
                  </div>
                  <h3 className="text-lg font-bold mb-2">Regulatory Library Access</h3>
                  <p className="text-gray-600 text-sm">
                    Access to comprehensive global regulatory frameworks and compliance templates.
                  </p>
                  <div className="mt-4">
                    <Badge className="bg-purple-100 text-purple-800 hover:bg-purple-200">Premium</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
        
        {/* Support */}
        <section className="mt-12">
          <Card className="bg-gradient-to-br from-indigo-900 to-blue-900 text-white border-none overflow-hidden shadow-lg">
            <CardContent className="p-8">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                <div>
                  <h3 className="text-2xl font-bold mb-2">Need Assistance with Reporting?</h3>
                  <p className="text-blue-100 mb-6 max-w-lg">
                    Our team of crypto compliance experts is available to help you create comprehensive due diligence reports that meet regulatory requirements.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <button className="inline-flex items-center gap-2 bg-white text-blue-900 px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors">
                      <Star className="h-4 w-4" />
                      <span>Schedule Consultation</span>
                    </button>
                    <button className="inline-flex items-center gap-2 bg-blue-800 border border-blue-700 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
                      <BookOpen className="h-4 w-4" />
                      <span>View Documentation</span>
                    </button>
                  </div>
                </div>
                <div className="bg-blue-800 bg-opacity-50 p-4 rounded-xl">
                  <Gift className="h-24 w-24 text-blue-300" />
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
      
      {/* Template Customization Modal */}
      <CustomizeTemplateModal 
        isOpen={isCustomizeTemplateModalOpen}
        onClose={handleCloseTemplateModal}
        onSave={handleSaveTemplate}
      />
    </div>
  );
}