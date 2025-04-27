"use client";

import { useCallback, useEffect, useState } from "react";
import { Upload, FileText, Search, Trash2, ExternalLink, Clock, AlertTriangle, Check, FileUp } from "lucide-react";
import { useDropzone } from 'react-dropzone';
import axios, { AxiosError, AxiosProgressEvent } from 'axios'; // Import AxiosProgressEvent
import { useRouter } from 'next/navigation';
/**/import { RiskDetailsModal } from "@/components/RiskDetailsModel";
import { fetchDocumentRisk } from "@/services/fetchDocumentRisk"; // Import


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
  is_public?: boolean; // Assuming this might be returned
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  // State to store the File object to be uploaded
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);
  // State to track upload progress { filename: percentage }
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadNotes, setUploadNotes] = useState('');
  const [isPublic, setIsPublic] = useState(false);
  //
  const [riskModalOpen, setRiskModalOpen] = useState(false);
  const [riskDetails, setRiskDetails] = useState({
  score: 0,
  category: "",
  factors: [] as string[],
  title: ""
  });
  // 
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

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    // Reset previous state for the new file
    setUploadError(null);
    setUploadProgress(0);
    setIsUploading(false);
    setUploadNotes('');
    setIsPublic(false);

    const file = acceptedFiles[0];
    console.log(`Selected file via dropzone: ${file.name}, size: ${file.size}, type: ${file.type}`);

    // Store the file object
    setFileToUpload(file);

    // Open the modal automatically
    setShowUploadModal(true);
  }, []); // Empty dependency array is correct here as it relies on setters which are stable

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    maxSize: 10485760, // 10MB
    multiple: false // Ensure only one file is accepted
  });

  const handleModalClose = () => {
    setShowUploadModal(false);
    setFileToUpload(null);
    setUploadProgress(0);
    setUploadNotes('');
    setIsPublic(false);
    setUploadError(null);
    setIsUploading(false);
  };

  const uploadFile = async () => {
    if (!fileToUpload) {
      setUploadError("No file selected for upload.");
      return;
    }

    const file = fileToUpload;

    // Reset upload error & set uploading state
    setUploadError(null);
    setIsUploading(true);
    setUploadProgress(0);

    // Create form data
    const formData = new FormData();
    formData.append('file', file); // Correctly appends the File object
    formData.append('notes', uploadNotes);
    formData.append('isPublic', isPublic.toString());

    try {
      // Upload the file with progress tracking
      const response = await axios.post('/api/documents/upload', formData, {
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          // Ensure progressEvent.total is defined and non-zero
          const total = progressEvent.total ?? file.size; // Fallback to file size
          const percentCompleted = total > 0
            ? Math.round((progressEvent.loaded * 100) / total)
            : 0; // Avoid division by zero

          setUploadProgress(percentCompleted);
        }
      });

      if (response.data.success) {
        // Clear the form, file state, and close the modal
        handleModalClose(); // Use the consolidated close handler
        // Fetch documents again to update the list
        fetchDocuments();
      } else {
        setUploadError(response.data.error || 'Upload failed');
        setUploadProgress(0); // Reset progress on server-side failure
        setIsUploading(false); // Allow retry
      }
    } catch (error: unknown) {
      console.error('Error uploading file:', error);
      const errorMessage = (error instanceof AxiosError && error.response?.data?.error)
        ? error.response.data.error
        : 'Failed to upload file. Please check the file type/size and network connection.';
      setUploadError(errorMessage);
      setUploadProgress(0); // Reset progress on network/request error
      setIsUploading(false); // Allow retry
    }
  };


  const deleteDocument = async (id: string) => {
    if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
      return;
    }

    try {
      setIsLoading(true); // Optional: Show loading indicator during delete
      const response = await axios.delete(`/api/documents/delete?id=${id}`);

      if (response.data.success) {
        // Remove document from the list visually
        setDocuments(prev => prev.filter(doc => doc.id !== id));
      } else {
        console.error('Failed to delete document:', response.data.error);
        alert(`Failed to delete document: ${response.data.error}`); // Show error to user
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      alert('An error occurred while trying to delete the document.'); // Show generic error
    } finally {
      setIsLoading(false);
    }
  };

  const openInQA = (documentId: string) => {
    router.push(`/dashboard/qa?documentId=${documentId}`);
  };

  const formatFileSize = (bytes?: number) => {
    if (bytes === undefined || bytes === null || bytes < 0) return 'N/A';
    if (bytes === 0) return '0 B';

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
        return <Clock className="h-5 w-5 text-yellow-500" aria-label="Pending" />;
      case 'processing':
        return <Clock className="h-5 w-5 text-blue-500 animate-pulse" aria-label="Processing" />;
      case 'completed':
        return <Check className="h-5 w-5 text-green-500" aria-label="Completed" />;
      case 'failed':
        return <AlertTriangle className="h-5 w-5 text-red-500" aria-label="Failed" />;
      default:
        return <Clock className="h-5 w-5 text-gray-500" aria-label="Unknown status" />;
    }
  };

  // Filter documents based on search query and status filter
  const filteredDocuments = documents.filter(doc => {
    const lowerSearchQuery = searchQuery.toLowerCase();
    const matchesSearch = searchQuery === '' ||
      doc.title.toLowerCase().includes(lowerSearchQuery) ||
      (doc.notes && doc.notes.toLowerCase().includes(lowerSearchQuery));

    const matchesStatus = statusFilter === '' || doc.processing_status === statusFilter;

    return matchesSearch && matchesStatus;
  });

  return (
    <div className="flex-1 p-4 md:p-8 pt-6">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-2">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">Document Analysis</h2>
          <p className="text-muted-foreground text-sm md:text-base">
            Upload, manage, and analyze crypto fund documentation.
          </p>
        </div>
        <div className="flex gap-4 w-full md:w-auto justify-end">
          <button
            className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 text-sm"
            onClick={() => setShowUploadModal(true)} // Opens the modal, onDrop will handle file selection later
          >
            <Upload size={18} />
            Upload Document
          </button>
        </div>
      </div>

      {/* Search and Filter Section */}
      <div className="mt-6 md:mt-8 flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
          <input
            type="text"
            placeholder="Search documents by title or notes..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            aria-label="Search documents"
          />
        </div>
        <select
          className="px-4 py-2 border rounded-lg text-sm bg-white focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          aria-label="Filter by status"
        >
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="processing">Processing</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      {/* Documents Grid / Loading / Empty State */}
      <div className="mt-6 md:mt-8">
        {isLoading && documents.length === 0 ? ( // Show spinner only on initial load
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : filteredDocuments.length > 0 ? (
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {filteredDocuments.map((doc) => (
              <div key={doc.id} className="bg-white p-4 rounded-lg shadow-sm border hover:shadow-md transition flex flex-col justify-between">
                <div> {/* Content wrapper */}
                  <div className="flex items-start gap-3 mb-3">
                    <FileText className="h-6 w-6 text-blue-500 flex-shrink-0 mt-1" />
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold truncate text-base" title={doc.title}>{doc.title}</h3>
                      <p className="text-xs text-gray-500">
                        {doc.file_type?.toUpperCase() || 'DOC'} • {formatFileSize(doc.file_size)}
                      </p>
                    </div>
                  </div>
                  <div className="space-y-1.5 text-sm mb-3">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-gray-600">Status:</span>
                      <span className="flex items-center gap-1">
                        {getStatusIcon(doc.processing_status)}
                        <span className="capitalize">{doc.processing_status}</span>
                      </span>
                    </div>

                    {doc.crypto_entities && doc.crypto_entities.length > 0 && (
                      <div className="text-xs">
                        <span className="font-medium text-gray-600">Entities:</span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {doc.crypto_entities.slice(0, 3).map((entity, idx) => (
                            <span key={idx} className="px-1.5 py-0.5 bg-blue-100 text-blue-800 rounded-full">
                              {entity}
                            </span>
                          ))}
                          {doc.crypto_entities.length > 3 && (
                            <span className="px-1.5 py-0.5 bg-gray-100 text-gray-800 rounded-full">
                              +{doc.crypto_entities.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {doc.risk_score !== undefined && doc.risk_score !== null && (
                      <div className="flex justify-between items-center">
                        <span className="font-medium text-gray-600">Risk Score:</span>
                        <span>{doc.risk_score.toFixed(1)}</span>
                      </div>
                    )}

                    <div className="flex justify-between items-center">
                        <span className="font-medium text-gray-600">Uploaded:</span>
                        <span className="text-gray-500">
                        {new Date(doc.upload_date).toLocaleDateString()}
                        </span>
                    </div>

                    {doc.notes && (
                      <div className="text-xs mt-1.5">
                        <span className="font-medium text-gray-600">Notes:</span>
                        <p className="text-gray-500 mt-0.5 line-clamp-2" title={doc.notes}>{doc.notes}</p>
                      </div>
                    )}
                  </div>
                </div> {/* End Content wrapper */}

                {/* Actions */}
                <div className="flex gap-2 mt-auto pt-3 border-t border-gray-100">
                  {/**/}
                  <button
                    onClick={async () => {
                      try {
                        const data = await fetchDocumentRisk(doc.id); // Use the new function

                        setRiskDetails({
                          score: data.risk_score,
                          category: data.risk_category,
                          factors: data.risk_factors,
                          title: data.title
                        });

                        setRiskModalOpen(true);
                      } catch (error) {
                        console.error("Could not fetch risk data:", error);
                        alert("Could not fetch risk data. Please check server logs.");
                      }
                    }}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800"

                  >
                    📊 View Risk
                  </button>
                   {/**/ }
                  <button
                    onClick={() => openInQA(doc.id)}
                    className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={doc.processing_status !== 'completed'}
                    title={doc.processing_status === 'completed' ? "Ask questions about this document" : "Document must be processed first"}
                  >
                    <ExternalLink size={14} />
                    Ask AI
                  </button>
                  <button
                    onClick={() => deleteDocument(doc.id)}
                    className="flex items-center gap-1 text-xs text-red-600 hover:text-red-800 ml-auto"
                    title="Delete document"
                  >
                    <Trash2 size={14} />
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 bg-gray-50/50 rounded-xl border border-dashed">
            <FileText className="h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-600">
                {searchQuery || statusFilter ? 'No matching documents' : 'No documents yet'}
            </h3>
            <p className="text-gray-500 mt-1 text-sm text-center px-4">
              {searchQuery || statusFilter ?
                'Try adjusting your search or filters.' :
                'Upload your first document to get started.'}
            </p>
            {!searchQuery && !statusFilter && ( // Show upload button only if not filtering
                 <button
                    className="mt-4 flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 text-sm"
                    onClick={() => setShowUploadModal(true)}
                >
                    <Upload size={18} />
                    Upload Document
                 </button>
            )}
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-60 z-50 p-4 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-lg max-w-md w-full p-6 relative">
            <button onClick={handleModalClose} className="absolute top-3 right-3 text-gray-400 hover:text-gray-600" aria-label="Close modal">×</button>
            <h2 className="text-xl font-semibold mb-5">Upload Document</h2>

            {/* Dropzone Area */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-6 mb-4 text-center cursor-pointer transition-colors duration-200 ease-in-out
                ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
                ${fileToUpload ? 'bg-gray-50 border-gray-300' : ''}`}
            >
              <input {...getInputProps()} />

              {fileToUpload ? (
                // Show selected file and progress
                <div>
                  <div className="mb-3 text-left">
                    <div className="flex items-center mb-1">
                      <FileUp className="mr-2 h-5 w-5 text-blue-500 flex-shrink-0" />
                      <span className="text-sm font-medium truncate" title={fileToUpload.name}>{fileToUpload.name}</span>
                    </div>
                    {isUploading && (
                        <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full transition-width duration-300 ease-out"
                          style={{ width: `${uploadProgress}%` }}
                        ></div>
                      </div>
                    )}
                    {!isUploading && uploadProgress === 0 && ( // Ready state
                        <p className="text-xs text-gray-500 mt-1">Ready to upload.</p>
                    )}
                  </div>
                   <p className="text-xs text-gray-500">Click or drag another file to replace.</p>
                </div>
              ) : (
                // Show upload prompt
                <div>
                  <Upload className="h-10 w-10 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-600 text-sm">Drag & drop file here</p>
                  <p className="text-xs text-gray-500 mt-1">or click to select</p>
                  <p className="text-xs text-gray-400 mt-2">PDF, DOCX, TXT (Max 10MB)</p>
                </div>
              )}
            </div>

            {/* Error message */}
            {uploadError && (
              <div className="bg-red-50 text-red-700 p-3 rounded-lg mb-4 text-sm border border-red-200">
                <p className="font-medium">Upload Failed</p>
                <p>{uploadError}</p>
              </div>
            )}

            {/* Form fields (only show if a file is selected) */}
            {fileToUpload && !isUploading && (
                <>
                 <div className="mb-4">
                    <label htmlFor="uploadNotes" className="block text-sm font-medium text-gray-700 mb-1">
                        Notes (optional)
                    </label>
                    <textarea
                        id="uploadNotes"
                        className="w-full border rounded-lg p-2 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                        rows={2}
                        value={uploadNotes}
                        onChange={(e) => setUploadNotes(e.target.value)}
                        placeholder="Add notes about this document..."
                    ></textarea>
                    </div>

                    <div className="mb-5">
                    <label className="flex items-center cursor-pointer">
                        <input
                        type="checkbox"
                        className="h-4 w-4 rounded text-blue-600 focus:ring-blue-500 border-gray-300 mr-2"
                        checked={isPublic}
                        onChange={(e) => setIsPublic(e.target.checked)}
                        />
                        <span className="text-sm text-gray-700">Make publicly accessible</span>
                        {/* Add Tooltip or Info icon here if needed */}
                    </label>
                    </div>
                </>
            )}


            {/* Actions */}
            <div className="flex justify-end gap-3 pt-4 border-t border-gray-200">
              <button
                className="px-4 py-2 rounded-lg border text-sm hover:bg-gray-50"
                onClick={handleModalClose}
                disabled={isUploading}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed"
                onClick={uploadFile}
                disabled={!fileToUpload || isUploading}
              >
                {isUploading ? `Uploading (${uploadProgress}%)` : 'Upload File'}
              </button>
            </div>
          </div>
        </div>
      )}
      <RiskDetailsModal
        isOpen={riskModalOpen}
        onClose={() => setRiskModalOpen(false)}
        riskScore={riskDetails.score}
        riskCategory={riskDetails.category}
        riskFactors={riskDetails.factors}
        title={riskDetails.title}
      />

      {riskModalOpen && <div className="fixed top-10 left-10 bg-red-500 text-white p-4 z-50">Risk Modal is open</div>}

    </div>
  );
}