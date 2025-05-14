// src/services/templates-service.ts
export interface ReportTemplate {
  id: string;
  name: string;
  type: string;
  branding: {
    useCompanyLogo: boolean;
    customColors: boolean;
    useFooter: boolean;
    applyWatermark: boolean;
  };
  colors: {
    primary: string;
    secondary: string;
    accent: string;
  };
  sections: string[];
  layout: {
    slidesPerSection: number;
    includeTableOfContents: boolean;
    includePageNumbers: boolean;
    chartStyle: string;
  };
  logo?: string | null;
}

// Get all templates
export async function getTemplates(): Promise<ReportTemplate[]> {
  // In a real app, you would fetch from your backend
  // For demo, we'll use localStorage
  try {
    const templates = localStorage.getItem('report_templates');
    return templates ? JSON.parse(templates) : [];
  } catch (error) {
    console.error('Error fetching templates:', error);
    return [];
  }
}

// Save a template
export async function saveTemplate(template: Omit<ReportTemplate, 'id'>): Promise<ReportTemplate> {
  try {
    // Get existing templates
    const templates = await getTemplates();
    
    // Create new template with ID
    const newTemplate: ReportTemplate = {
      ...template,
      id: crypto.randomUUID(),
    };
    
    // Add to templates and save
    templates.push(newTemplate);
    localStorage.setItem('report_templates', JSON.stringify(templates));
    
    return newTemplate;
  } catch (error) {
    console.error('Error saving template:', error);
    throw error;
  }
}

// Delete a template
export async function deleteTemplate(id: string): Promise<void> {
  try {
    // Get existing templates
    const templates = await getTemplates();
    
    // Filter out the template to delete
    const updatedTemplates = templates.filter(t => t.id !== id);
    
    // Save updated templates
    localStorage.setItem('report_templates', JSON.stringify(updatedTemplates));
  } catch (error) {
    console.error('Error deleting template:', error);
    throw error;
  }
}