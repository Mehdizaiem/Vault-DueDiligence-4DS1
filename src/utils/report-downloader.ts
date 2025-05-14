// src/utils/report-downloader.ts
import { saveAs } from 'file-saver';

interface ReportDownloadOptions {
  title: string;
  type: string;
  reportId: string;
  report: any; // Full report data
}

/**
 * Download a report as a PPTX file
 * This is a demo implementation that simulates downloading a PPTX file
 */
export const downloadReportAsPPTX = async (options: ReportDownloadOptions): Promise<boolean> => {
  const { title, type, reportId, report } = options;
  
  try {
    // For demo purposes, we'll simulate an API call with a timeout
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Create a simple text file with JSON data that represents what would be in the PPTX
    // In a real implementation, this would be a PPTX file generated on the server
    const reportData = {
      title: report.title,
      type: report.type,
      risk_score: report.risk_score,
      compliance_score: report.compliance_score,
      entities: report.entities,
      created_at: report.created_at,
      author: report.author,
      sections: [
        "Executive Summary",
        "Risk Assessment",
        "Compliance Status",
        "Entity Analysis",
        "Recommendations"
      ]
    };
    
    // Convert to JSON string with pretty formatting
    const jsonContent = JSON.stringify(reportData, null, 2);
    
    // Create a Blob with the JSON content
    // Note: In a real app, this would be the binary content of a PPTX file
    const mockContent = new Blob(
      [jsonContent],
      { type: 'application/json' }
    );
    
    // Generate a filename based on the report details
    // For demo purposes, we'll use .pptx extension even though it's actually JSON
    const filename = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_${reportId}.pptx`;
    
    // Save the file
    saveAs(mockContent, filename);
    
    return true;
  } catch (error) {
    console.error('Error downloading report:', error);
    return false;
  }
};

/**
 * In a real application, you would have multiple download formats
 * Here we've added PDF as an example
 */
export const downloadReportAsPDF = async (options: ReportDownloadOptions): Promise<boolean> => {
  const { title, type, reportId, report } = options;
  
  try {
    // Similar implementation to PPTX but for PDF
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const reportData = {
      title: report.title,
      type: report.type,
      risk_score: report.risk_score,
      compliance_score: report.compliance_score,
      entities: report.entities,
      created_at: report.created_at,
      author: report.author
    };
    
    const jsonContent = JSON.stringify(reportData, null, 2);
    
    const mockContent = new Blob(
      [jsonContent],
      { type: 'application/json' }
    );
    
    const filename = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_${reportId}.pdf`;
    
    saveAs(mockContent, filename);
    
    return true;
  } catch (error) {
    console.error('Error downloading report:', error);
    return false;
  }
};