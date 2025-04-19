"use client";

import { useCallback, useEffect, useState } from "react";
import { Upload, FileText, Search,  Trash2, ExternalLink, Clock, AlertTriangle, Check, FileUp } from "lucide-react";
import { useDropzone } from 'react-dropzone';
import axios, { AxiosError } from 'axios';
import { useRouter } from 'next/navigation';

interface Document {
  id: string;
  title: string;
  file_type?: string;
  file_size?: number;
  upload_date: string;
  processing_status: string;
  risk_score?: number;
  notes?: string;
  crypto_entities?: string[];
  risk_factors?: string[];
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [uploadingFiles, setUploadingFiles] = useState<{ [key: string]: number }>({});
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadNotes, setUploadNotes] = useState('');
  const [isPublic, setIsPublic] = useState(false);
  const router = useRouter();

  // Fetch documents when component mounts
  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get('/api/documents/list');
      if (response.data.success) {
        setDocuments(response.data.documents);
      } else {
        console.error('Failed to fetch documents:', response.data.error);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setUploadError(null);
    setShowUploadModal(true);
    
    // Store files in a state for later upload
    const newUploadingFiles: { [key: string]: number } = {};
    acceptedFiles.forEach(file => {
      newUploadingFiles[file.name] = 0;
    });
    
    setUploadingFiles(newUploadingFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxSize: 10485760, // 10MB
    multiple: false
  });

  const uploadFile = async () => {
    const files = Object.keys(uploadingFiles).map(
      name => new File(
        [new Blob()], 
        name, 
        { type: name.endsWith('.pdf') ? 'application/pdf' : 
                name.endsWith('.docx') ? 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' : 
                'text/plain' }
      )
    );
    
    if (files.length === 0) return;
    
    const file = files[0];
    
    // Reset upload error
    setUploadError(null);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('notes', uploadNotes);
    formData.append('isPublic', isPublic.toString());
    
    try {
      // Upload the file with progress tracking
      const response = await axios.post('/api/documents/upload', formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          );
          
          setUploadingFiles(prev => ({
            ...prev,
            [file.name]: percentCompleted
          }));
        }
      });
      
      if (response.data.success) {
        // Clear the form and close the modal
        setUploadingFiles({});
        setUploadNotes('');
        setIsPublic(false);
        setShowUploadModal(false);
        
        // Fetch documents again to update the list
        fetchDocuments();
      } else {
        setUploadError(response.data.error || 'Upload failed');
      }
    } // Then in the catch block:
    catch (error: unknown) {
      console.error('Error uploading file:', error);
      if (error instanceof AxiosError && error.response?.data?.error) {
        setUploadError(error.response.data.error);
      } else {
        setUploadError('Failed to upload file');
      }
    }
  };

  const deleteDocument = async (id: string) => {
    if (!confirm('Are you sure you want to delete this document?')) {
      return;
    }
    
    try {
      const response = await axios.delete(`/api/documents/delete?id=${id}`);
      
      if (response.data.success) {
        // Remove document from the list
        setDocuments(prev => prev.filter(doc => doc.id !== id));
      } else {
        console.error('Failed to delete document:', response.data.error);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  const openInQA = (documentId: string) => {
    router.push(`/dashboard/qa?documentId=${documentId}`);
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return 'Unknown';
    
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'processing':
        return <Clock className="h-5 w-5 text-blue-500 animate-pulse" />;
      case 'completed':
        return <Check className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" />;
    }
  };

  // Filter documents based on search query and status filter
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = searchQuery === '' || 
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (doc.notes && doc.notes.toLowerCase().includes(searchQuery.toLowerCase()));
      
    const matchesStatus = statusFilter === '' || doc.processing_status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

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
          <button 
            className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800"
            onClick={() => setShowUploadModal(true)}
          >
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
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 border rounded-lg"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="processing">Processing</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      {/* Documents Grid */}
      <div className="mt-8">
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : filteredDocuments.length > 0 ? (
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {filteredDocuments.map((doc) => (
              <div key={doc.id} className="bg-white p-6 rounded-xl shadow-sm border hover:shadow-md transition">
                <div className="flex items-center gap-3 mb-4">
                  <FileText className="h-8 w-8 text-blue-500" />
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold truncate">{doc.title}</h3>
                    <p className="text-sm text-gray-500">
                      {doc.file_type?.toUpperCase() || 'Document'} • {formatFileSize(doc.file_size)}
                    </p>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <span className="font-medium">Status:</span>
                    <span className="flex items-center gap-1">
                      {getStatusIcon(doc.processing_status)}
                      <span className="capitalize">{doc.processing_status}</span>
                    </span>
                  </div>
                  
                  {doc.crypto_entities && doc.crypto_entities.length > 0 && (
                    <div className="text-sm">
                      <span className="font-medium">Entities:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {doc.crypto_entities.slice(0, 3).map((entity, idx) => (
                          <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                            {entity}
                          </span>
                        ))}
                        {doc.crypto_entities.length > 3 && (
                          <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs">
                            +{doc.crypto_entities.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {doc.risk_score !== undefined && (
                    <div className="text-sm">
                      <span className="font-medium">Risk Score:</span>
                      <span className="ml-2">{doc.risk_score.toFixed(1)}</span>
                    </div>
                  )}
                  
                  <div className="text-sm">
                    <span className="font-medium">Uploaded:</span>
                    <span className="ml-2 text-gray-500">
                      {new Date(doc.upload_date).toLocaleDateString()}
                    </span>
                  </div>
                  
                  {doc.notes && (
                    <div className="text-sm mt-2">
                      <span className="font-medium">Notes:</span>
                      <p className="text-gray-500 mt-1 line-clamp-2">{doc.notes}</p>
                    </div>
                  )}
                </div>
                
                {/* Actions */}
                <div className="flex gap-2 mt-4 pt-3 border-t border-gray-100">
                  <button 
                    onClick={() => openInQA(doc.id)}
                    className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800"
                  >
                    <ExternalLink size={14} />
                    Ask Questions
                  </button>
                  <button 
                    onClick={() => deleteDocument(doc.id)}
                    className="flex items-center gap-1 text-sm text-red-600 hover:text-red-800 ml-auto"
                  >
                    <Trash2 size={14} />
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 bg-gray-50 rounded-xl border border-dashed">
            <FileText className="h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-600">No documents found</h3>
            <p className="text-gray-500 mt-1">
              {searchQuery || statusFilter ? 
                'Try adjusting your search or filters' : 
                'Upload your first document to get started'}
            </p>
            <button 
              className="mt-4 flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
              onClick={() => setShowUploadModal(true)}
            >
              <Upload size={18} />
              Upload Document
            </button>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white rounded-xl shadow-lg max-w-md w-full p-6">
            <h2 className="text-xl font-bold mb-4">Upload Document</h2>
            
            {/* Dropzone */}
            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-lg p-6 mb-4 text-center cursor-pointer
                ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-500'}
                ${Object.keys(uploadingFiles).length > 0 ? 'bg-gray-50' : ''}`}
            >
              <input {...getInputProps()} />
              
              {Object.keys(uploadingFiles).length > 0 ? (
                <div>
                  {Object.entries(uploadingFiles).map(([name, progress]) => (
                    <div key={name} className="mb-3">
                      <div className="flex items-center mb-1">
                        <FileUp className="mr-2 h-5 w-5 text-blue-500" />
                        <span className="text-sm font-medium truncate">{name}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${progress}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div>
                  <Upload className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-600">Drag & drop a document here, or click to select</p>
                  <p className="text-sm text-gray-500 mt-1">Supported formats: PDF, DOCX, TXT (Max 10MB)</p>
                </div>
              )}
            </div>

            {/* Error message */}
            {uploadError && (
              <div className="bg-red-50 text-red-600 p-3 rounded-lg mb-4">
                {uploadError}
              </div>
            )}
            
            {/* Form fields */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Notes (optional)
              </label>
              <textarea
                className="w-full border rounded-lg p-2"
                rows={3}
                value={uploadNotes}
                onChange={(e) => setUploadNotes(e.target.value)}
                placeholder="Add notes about this document..."
              ></textarea>
            </div>
            
            <div className="mb-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="rounded text-blue-600 mr-2"
                  checked={isPublic}
                  onChange={(e) => setIsPublic(e.target.checked)}
                />
                <span className="text-sm text-gray-700">Make this document visible to other team members</span>
              </label>
            </div>
            
            {/* Actions */}
            <div className="flex justify-end gap-3">
              <button
                className="px-4 py-2 rounded-lg border hover:bg-gray-50"
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadingFiles({});
                  setUploadNotes('');
                  setIsPublic(false);
                  setUploadError(null);
                }}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300"
                onClick={uploadFile}
                disabled={Object.keys(uploadingFiles).length === 0}
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}