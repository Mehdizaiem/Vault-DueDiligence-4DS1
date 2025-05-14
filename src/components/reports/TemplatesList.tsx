// src/components/reports/TemplatesList.tsx
import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { FileText, Edit, Trash2, Copy } from 'lucide-react';
import { deleteTemplate } from '@/services/templates-service';

interface TemplatesListProps {
  templates: any[];
  onRefresh: () => void;
  onEdit: (template: any) => void;
}

export default function TemplatesList({ templates, onRefresh, onEdit }: TemplatesListProps) {
  const [loading, setLoading] = useState(false);
  
  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this template?')) return;
    
    setLoading(true);
    try {
      await deleteTemplate(id);
      onRefresh();
    } catch (error) {
      console.error('Error deleting template:', error);
      alert('Failed to delete template');
    } finally {
      setLoading(false);
    }
  };
  
  const handleDuplicate = async (template: any) => {
    try {
      // Create a duplicate by removing the ID
      const templateData = { ...template };
      delete templateData.id;
      templateData.name = `${templateData.name} (Copy)`;
      
      onEdit(templateData);
    } catch (error) {
      console.error('Error duplicating template:', error);
      alert('Failed to duplicate template');
    }
  };
  
  if (templates.length === 0) {
    return (
      <div className="text-center py-8 bg-gray-50 border border-dashed rounded-lg">
        <FileText className="h-12 w-12 text-gray-400 mx-auto mb-3" />
        <h3 className="text-lg font-medium text-gray-700 mb-2">No custom templates yet</h3>
        <p className="text-gray-500 mb-4">Create your first template to get started</p>
      </div>
    );
  }
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {templates.map((template) => (
        <Card key={template.id} className="hover:shadow-md transition-shadow duration-200">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">{template.name}</CardTitle>
            <p className="text-sm text-gray-500 mt-1 capitalize">{template.type}</p>
          </CardHeader>
          <CardContent>
            <div className="text-sm space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-500">Sections:</span>
                <span>{template.sections.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Custom Branding:</span>
                <span>{template.branding.customColors ? 'Yes' : 'No'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Slides Per Section:</span>
                <span>{template.layout.slidesPerSection}</span>
              </div>
            </div>
          </CardContent>
          <CardFooter className="border-t pt-3 flex justify-between">
            <button
              onClick={() => handleDuplicate(template)}
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
            >
              <Copy size={14} />
              <span>Duplicate</span>
            </button>
            <div className="flex gap-3">
              <button
                onClick={() => onEdit(template)}
                className="text-sm text-gray-600 hover:text-gray-800 flex items-center gap-1"
              >
                <Edit size={14} />
                <span>Edit</span>
              </button>
              <button
                onClick={() => handleDelete(template.id)}
                className="text-sm text-red-600 hover:text-red-800 flex items-center gap-1"
                disabled={loading}
              >
                <Trash2 size={14} />
                <span>Delete</span>
              </button>
            </div>
          </CardFooter>
        </Card>
      ))}
    </div>
  );
}