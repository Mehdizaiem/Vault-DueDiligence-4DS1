// src/components/reports/CustomizeTemplateModal.tsx
// Make sure CustomizeTemplateForm.tsx exists in the same folder, or update the path if it's elsewhere

import CustomizeTemplateForm from "./CustomizeTemplateForm";

interface CustomizeTemplateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (template: any) => void;
}

const CustomizeTemplateModal = ({ isOpen, onClose, onSave }: CustomizeTemplateModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 overflow-auto">
      <div className="max-w-5xl w-full max-h-[90vh] overflow-auto bg-white rounded-xl shadow-2xl">
        <CustomizeTemplateForm onClose={onClose} onSave={onSave} />
      </div>
    </div>
  );
};

export default CustomizeTemplateModal;