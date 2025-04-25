interface TimeFrameSelectorProps {
  selected: '90d' | '180d' | '365d';
  onChange: (value: '90d' | '180d' | '365d') => void;
}

export default function TimeFrameSelector({ selected, onChange }: TimeFrameSelectorProps) {
  return (
    <div className="flex items-center gap-0 bg-white rounded-lg border overflow-hidden shadow-sm">
      <button 
        className={`px-3 py-2 text-sm font-medium transition-colors ${
          selected === '90d' 
            ? 'bg-black text-white' 
            : 'text-gray-600 hover:bg-gray-50'
        }`}
        onClick={() => onChange('90d')}
      >
        90d
      </button>
      <button 
        className={`px-3 py-2 text-sm font-medium transition-colors ${
          selected === '180d' 
            ? 'bg-black text-white' 
            : 'text-gray-600 hover:bg-gray-50'
        }`}
        onClick={() => onChange('180d')}
      >
        180d
      </button>
      <button 
        className={`px-3 py-2 text-sm font-medium transition-colors ${
          selected === '365d' 
            ? 'bg-black text-white' 
            : 'text-gray-600 hover:bg-gray-50'
        }`}
        onClick={() => onChange('365d')}
      >
        365d
      </button>
    </div>
  );
}