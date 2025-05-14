// src/components/reports/CustomizeTemplateModal.tsx
import CustomizeTemplateForm from "./CustomizeTemplateForm";

interface CustomizeTemplateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (template: any) => void;
}

const CustomizeTemplateModal = ({ isOpen, onClose, onSave }: CustomizeTemplateModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="max-w-5xl w-full h-[90vh] overflow-auto bg-white rounded-xl shadow-2xl">
        <CustomizeTemplateForm onClose={onClose} onSave={onSave} />
      </div>
    </div>
  );
};

export default CustomizeTemplateModal;