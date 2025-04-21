"""
Template management for report generation
"""
import os
from pathlib import Path
from typing import Dict, Optional

class TemplateManager:
    """Manages PowerPoint templates for report generation."""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent / 'templates'
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load available templates."""
        templates = {}
        
        # Create template directory if it doesn't exist
        self.template_dir.mkdir(exist_ok=True)
        
        # Look for .pptx files in template directory
        for file_path in self.template_dir.glob('*.pptx'):
            template_name = file_path.stem
            templates[template_name] = str(file_path)
        
        return templates
    
    def get_template_path(self, template_name: str) -> Optional[str]:
        """Get path to a specific template."""
        return self.templates.get(template_name)
    
    def list_templates(self) -> list:
        """List available templates."""
        return list(self.templates.keys())
    
    def get_default_template_path(self) -> Optional[str]:
        """Get path to the default template."""
        return self.templates.get('default_template')