// src/components/reports/CustomizeTemplateForm.tsx

import { useState } from 'react';
import {
  FileText, 
  Settings,
  Check,
  ChevronDown,
  Layout,
  Image,
  Target,
  Clock,
  Save,
  FileSymlink,
  Upload,
  Paintbrush,
  Sliders,
  Layers,
  Edit3,
  AlignLeft,
  Package
} from 'lucide-react';

interface CustomizeTemplateFormProps {
  onClose: () => void;
  onSave: (template: any) => void;
  initialTemplate?: any;
}

const CustomizeTemplateForm = ({ onClose, onSave, initialTemplate }: CustomizeTemplateFormProps) => {
  const [templateName, setTemplateName] = useState(initialTemplate?.name || '');
  const [templateType, setTemplateType] = useState(initialTemplate?.type || 'comprehensive');
  const [brandingOptions, setBrandingOptions] = useState({
    useCompanyLogo: initialTemplate?.branding?.useCompanyLogo ?? true,
    customColors: initialTemplate?.branding?.customColors ?? false,
    useFooter: initialTemplate?.branding?.useFooter ?? true,
    applyWatermark: initialTemplate?.branding?.applyWatermark ?? false
  });
  
  const [primaryColor, setPrimaryColor] = useState(initialTemplate?.colors?.primary || '#4c6bff');
  const [secondaryColor, setSecondaryColor] = useState(initialTemplate?.colors?.secondary || '#10b981');
  const [accentColor, setAccentColor] = useState(initialTemplate?.colors?.accent || '#f97316');
  
  const [selectedSections, setSelectedSections] = useState(initialTemplate?.sections || [
    'executive_summary',
    'risk_assessment',
    'compliance_status',
    'entity_analysis',
    'performance_metrics'
  ]);
  
  const availableSections = [
    { id: 'executive_summary', name: 'Executive Summary', icon: FileText, required: true },
    { id: 'risk_assessment', name: 'Risk Assessment', icon: Target, required: false },
    { id: 'compliance_status', name: 'Compliance Status', icon: Check, required: false },
    { id: 'entity_analysis', name: 'Entity Analysis', icon: Layers, required: false },
    { id: 'performance_metrics', name: 'Performance Metrics', icon: Sliders, required: false },
    { id: 'historical_comparison', name: 'Historical Comparison', icon: Clock, required: false },
    { id: 'market_sentiment', name: 'Market Sentiment', icon: ChevronDown, required: false },
    { id: 'recommendations', name: 'Recommendations', icon: Edit3, required: false },
    { id: 'regulatory_highlights', name: 'Regulatory Highlights', icon: AlignLeft, required: false },
    { id: 'appendix', name: 'Appendix & References', icon: FileSymlink, required: false }
  ];
  
  // Layout options
  const [layoutOptions, setLayoutOptions] = useState({
    slidesPerSection: initialTemplate?.layout?.slidesPerSection || 2,
    includeTableOfContents: initialTemplate?.layout?.includeTableOfContents ?? true,
    includePageNumbers: initialTemplate?.layout?.includePageNumbers ?? true,
    chartStyle: initialTemplate?.layout?.chartStyle || 'modern'
  });
  
  // Custom Logo upload
  const [logoFile, setLogoFile] = useState<File | null>(null);
  
  const handleUploadLogo = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setLogoFile(file);
    }
  };
  
  const toggleSection = (sectionId: string) => {
    const section = availableSections.find(s => s.id === sectionId);
    
    // Don't allow removing required sections
    if (section && section.required && selectedSections.includes(sectionId)) {
      return;
    }
    
    setSelectedSections((prev: string[]) => 
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };
  
  const reorderSection = (sectionId: string, direction: 'up' | 'down') => {
    const currentIndex = selectedSections.indexOf(sectionId);
    if (currentIndex === -1) return;
    
    const newSections = [...selectedSections];
    
    if (direction === 'up' && currentIndex > 0) {
      // Swap with previous item
      [newSections[currentIndex], newSections[currentIndex - 1]] = 
      [newSections[currentIndex - 1], newSections[currentIndex]];
    } else if (direction === 'down' && currentIndex < selectedSections.length - 1) {
      // Swap with next item
      [newSections[currentIndex], newSections[currentIndex + 1]] = 
      [newSections[currentIndex + 1], newSections[currentIndex]];
    }
    
    setSelectedSections(newSections);
  };
  
  const handleSave = () => {
    // Validate template name
    if (!templateName.trim()) {
      alert('Please enter a template name');
      return;
    }
    
    // Create template object
    const template = {
      ...(initialTemplate?.id ? { id: initialTemplate.id } : {}), // Preserve ID if editing
      name: templateName,
      type: templateType,
      branding: brandingOptions,
      colors: {
        primary: primaryColor,
        secondary: secondaryColor,
        accent: accentColor
      },
      sections: selectedSections,
      layout: layoutOptions,
      logo: logoFile ? logoFile.name : (initialTemplate?.logo || null)
    };
    
    // Pass to parent component
    onSave(template);
  };
  
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-purple-100">
            <Settings className="h-6 w-6 text-purple-600" />
          </div>
          <h2 className="text-2xl font-bold">{initialTemplate ? 'Edit Template' : 'Create New Template'}</h2>
        </div>
        <button 
          onClick={onClose}
          className="text-gray-500 hover:text-gray-800 transition-colors"
        >
          &times;
        </button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Basic Settings */}
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
              <FileText className="h-5 w-5 text-gray-500" />
              Template Information
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Template Name
                </label>
                <input
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="e.g., Quarterly Risk Assessment"
                  className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Base Template Type
                </label>
                <select
                  value={templateType}
                  onChange={(e) => setTemplateType(e.target.value)}
                  className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="comprehensive">Comprehensive Due Diligence</option>
                  <option value="compliance">Regulatory Compliance</option>
                  <option value="risk">Risk Assessment</option>
                  <option value="comparative">Comparative Analysis</option>
                  <option value="security">Security Audit</option>
                  <option value="blank">Blank Template</option>
                </select>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
              <Paintbrush className="h-5 w-5 text-gray-500" />
              Branding
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">
                  Use Company Logo
                </label>
                <div className="relative inline-block w-10 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={brandingOptions.useCompanyLogo}
                    onChange={() => setBrandingOptions({...brandingOptions, useCompanyLogo: !brandingOptions.useCompanyLogo})}
                    className="checked:bg-blue-500 outline-none focus:outline-none right-4 checked:right-0 duration-200 ease-in absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className={`block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer ${brandingOptions.useCompanyLogo ? "bg-blue-400" : ""}`} />
                </div>
              </div>
              
              {brandingOptions.useCompanyLogo && (
                <div className="mt-2">
                  <div className="flex items-center gap-2 mb-2">
                    <input 
                      type="file" 
                      id="logo-upload" 
                      className="hidden"
                      accept="image/*"
                      onChange={handleUploadLogo}
                    />
                    <label 
                      htmlFor="logo-upload"
                      className="flex items-center gap-2 text-sm px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg cursor-pointer"
                    >
                      <Upload size={16} />
                      {logoFile ? 'Change Logo' : initialTemplate?.logo ? 'Replace Logo' : 'Upload Logo'}
                    </label>
                    {(logoFile && (
                      <span className="text-xs text-gray-500">{logoFile.name}</span>
                    )) || (initialTemplate?.logo && !logoFile && (
                      <span className="text-xs text-gray-500">{initialTemplate.logo}</span>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">
                  Custom Color Scheme
                </label>
                <div className="relative inline-block w-10 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={brandingOptions.customColors}
                    onChange={() => setBrandingOptions({...brandingOptions, customColors: !brandingOptions.customColors})}
                    className="checked:bg-blue-500 outline-none focus:outline-none right-4 checked:right-0 duration-200 ease-in absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className={`block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer ${brandingOptions.customColors ? "bg-blue-400" : ""}`} />
                </div>
              </div>
              
              {brandingOptions.customColors && (
                <div className="grid grid-cols-3 gap-2 pt-2">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Primary</label>
                    <input 
                      type="color" 
                      value={primaryColor}
                      onChange={(e) => setPrimaryColor(e.target.value)}
                      className="w-full h-8 rounded cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Secondary</label>
                    <input 
                      type="color" 
                      value={secondaryColor}
                      onChange={(e) => setSecondaryColor(e.target.value)}
                      className="w-full h-8 rounded cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Accent</label>
                    <input 
                      type="color" 
                      value={accentColor}
                      onChange={(e) => setAccentColor(e.target.value)}
                      className="w-full h-8 rounded cursor-pointer"
                    />
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">
                  Include Page Footer
                </label>
                <div className="relative inline-block w-10 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={brandingOptions.useFooter}
                    onChange={() => setBrandingOptions({...brandingOptions, useFooter: !brandingOptions.useFooter})}
                    className="checked:bg-blue-500 outline-none focus:outline-none right-4 checked:right-0 duration-200 ease-in absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className={`block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer ${brandingOptions.useFooter ? "bg-blue-400" : ""}`} />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">
                  Apply Watermark
                </label>
                <div className="relative inline-block w-10 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={brandingOptions.applyWatermark}
                    onChange={() => setBrandingOptions({...brandingOptions, applyWatermark: !brandingOptions.applyWatermark})}
                    className="checked:bg-blue-500 outline-none focus:outline-none right-4 checked:right-0 duration-200 ease-in absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className={`block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer ${brandingOptions.applyWatermark ? "bg-blue-400" : ""}`} />
                </div>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
              <Layout className="h-5 w-5 text-gray-500" />
              Layout Options
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Slides Per Section
                </label>
                <select
                  value={layoutOptions.slidesPerSection}
                  onChange={(e) => setLayoutOptions({...layoutOptions, slidesPerSection: parseInt(e.target.value)})}
                  className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="1">1 (Minimal)</option>
                  <option value="2">2 (Standard)</option>
                  <option value="3">3 (Detailed)</option>
                  <option value="4">4 (Comprehensive)</option>
                  <option value="5">5 (Extended)</option>
                </select>
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">
                  Include Table of Contents
                </label>
                <div className="relative inline-block w-10 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={layoutOptions.includeTableOfContents}
                    onChange={() => setLayoutOptions({...layoutOptions, includeTableOfContents: !layoutOptions.includeTableOfContents})}
                    className="checked:bg-blue-500 outline-none focus:outline-none right-4 checked:right-0 duration-200 ease-in absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className={`block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer ${layoutOptions.includeTableOfContents ? "bg-blue-400" : ""}`} />
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">
                  Include Page Numbers
                </label>
                <div className="relative inline-block w-10 align-middle select-none">
                  <input
                    type="checkbox"
                    checked={layoutOptions.includePageNumbers}
                    onChange={() => setLayoutOptions({...layoutOptions, includePageNumbers: !layoutOptions.includePageNumbers})}
                    className="checked:bg-blue-500 outline-none focus:outline-none right-4 checked:right-0 duration-200 ease-in absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                  />
                  <label className={`block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer ${layoutOptions.includePageNumbers ? "bg-blue-400" : ""}`} />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Chart Style
                </label>
                <select
                  value={layoutOptions.chartStyle}
                  onChange={(e) => setLayoutOptions({...layoutOptions, chartStyle: e.target.value})}
                  className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="modern">Modern (Default)</option>
                  <option value="classic">Classic</option>
                  <option value="minimal">Minimal</option>
                  <option value="colorful">Colorful</option>
                  <option value="monochrome">Monochrome</option>
                </select>
              </div>
            </div>
          </div>
        </div>
        
        {/* Right column - Section Selection */}
        <div className="lg:col-span-2">
          <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
            <Layers className="h-5 w-5 text-gray-500" />
            Report Sections
          </h3>
          
          <div className="border rounded-lg overflow-hidden">
            <div className="bg-gray-50 p-3 border-b">
              <p className="text-sm text-gray-500">
                Drag and reorder sections. Required sections cannot be removed.
              </p>
            </div>
            
            <div className="max-h-[500px] overflow-y-auto">
              {availableSections.map((section) => {
                const isSelected = selectedSections.includes(section.id);
                const sectionIndex = selectedSections.indexOf(section.id);
                
                return (
                  <div 
                    key={section.id}
                    className={`p-3 border-b last:border-0 ${
                      isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'
                    } transition-colors duration-150`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSection(section.id)}
                          disabled={section.required && isSelected}
                          className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                        />
                        <div className={`p-1.5 rounded ${isSelected ? 'bg-blue-100' : 'bg-gray-100'}`}>
                          <section.icon className={`h-4 w-4 ${isSelected ? 'text-blue-600' : 'text-gray-500'}`} />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center">
                            <span className="font-medium text-gray-900">
                              {section.name}
                            </span>
                            {section.required && (
                              <span className="ml-2 text-xs font-medium px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded">
                                Required
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      
                      {isSelected && (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => reorderSection(section.id, 'up')}
                            disabled={sectionIndex === 0}
                            className={`p-1 rounded ${sectionIndex === 0 ? 'text-gray-300 cursor-not-allowed' : 'text-gray-500 hover:bg-gray-200'}`}
                          >
                            <ChevronDown className="h-4 w-4 transform rotate-180" />
                          </button>
                          <button
                            onClick={() => reorderSection(section.id, 'down')}
                            disabled={sectionIndex === selectedSections.length - 1}
                            className={`p-1 rounded ${sectionIndex === selectedSections.length - 1 ? 'text-gray-300 cursor-not-allowed' : 'text-gray-500 hover:bg-gray-200'}`}
                          >
                            <ChevronDown className="h-4 w-4" />
                          </button>
                          <span className="text-xs text-gray-500">
                            {sectionIndex + 1}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    {isSelected && (
                      <div className="mt-2 pl-10">
                        <div className="flex items-center text-xs text-gray-500">
                          <Layout className="h-3.5 w-3.5 mr-1.5" />
                          <span>
                            {layoutOptions.slidesPerSection} slide{layoutOptions.slidesPerSection !== 1 ? 's' : ''}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
      
      {/* Live Preview */}
      <div className="mt-8 pt-6 border-t">
        <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
          <Image className="h-5 w-5 text-gray-500" />
          Template Preview
        </h3>
        
        <div className="bg-gray-50 border rounded-lg p-4 flex items-center justify-center h-48">
          <div className="bg-white rounded border shadow-sm w-64 h-36 relative overflow-hidden">
            {/* Header */}
            <div 
              className="h-8 w-full flex items-center px-2"
              style={{ backgroundColor: brandingOptions.customColors ? primaryColor : '#4c6bff' }}
            >
              <div className="text-white text-xs font-medium truncate">
                {templateName || "Custom Template"}
              </div>
              {brandingOptions.useCompanyLogo && (
                <div className="absolute right-2 top-1 bg-white rounded w-6 h-6 flex items-center justify-center">
                  {logoFile || initialTemplate?.logo ? (
                    <span className="text-xs">LOGO</span>
                  ) : (
                    <Package className="h-3 w-3 text-gray-400" />
                  )}
                </div>
              )}
            </div>
            
            {/* Content */}
            <div className="p-2">
              <div className="h-4 w-3/4 bg-gray-200 rounded mb-2"></div>
              <div className="h-3 w-full bg-gray-200 rounded mb-1"></div>
              <div className="h-3 w-5/6 bg-gray-200 rounded mb-1"></div>
              <div className="h-3 w-4/6 bg-gray-200 rounded"></div>
            </div>
            
            {/* Chart */}
            <div 
              className="absolute bottom-2 right-2 w-16 h-12 rounded"
              style={{ backgroundColor: brandingOptions.customColors ? secondaryColor : '#10b981' }}
            >
            </div>
            
            {/* Watermark */}
            {brandingOptions.applyWatermark && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-gray-300 text-lg font-bold rotate-45 opacity-20">
                  VAULT
                </div>
              </div>
            )}
            
            {/* Footer */}
            {brandingOptions.useFooter && (
              <div className="absolute bottom-0 left-0 right-0 h-4 bg-gray-100 flex items-center justify-between px-2">
                <div className="h-2 w-12 bg-gray-300 rounded"></div>
                {layoutOptions.includePageNumbers && (
                  <div className="h-2 w-4 bg-gray-300 rounded"></div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Buttons */}
      <div className="mt-8 pt-4 border-t flex justify-end gap-3">
        <button
          onClick={onClose}
          className="px-4 py-2 border rounded-lg hover:bg-gray-50"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
        >
          <Save className="h-4 w-4" />
          Save Template
        </button>
      </div>
    </div>
  );
};

export default CustomizeTemplateForm;