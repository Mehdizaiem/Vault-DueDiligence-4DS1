import { Card } from "@/components/ui/card";
import { 
  FileText, 
  HelpCircle, 
  AlertTriangle,
  Users,
  DollarSign,
  Shield,
  Activity
} from "lucide-react";

const stats = [
  {
    label: 'Total Funds',
    value: '28',
    icon: Users,
    color: 'text-blue-500',
    bgColor: 'bg-blue-100'
  },
  {
    label: 'AUM',
    value: '$2.4B',
    icon: DollarSign,
    color: 'text-green-500',
    bgColor: 'bg-green-100'
  },
  {
    label: 'Risk Score',
    value: '94%',
    icon: Shield,
    color: 'text-violet-500',
    bgColor: 'bg-violet-100'
  },
  {
    label: 'Active Alerts',
    value: '7',
    icon: Activity,
    color: 'text-pink-500',
    bgColor: 'bg-pink-100'
  }
];

const features = [
  {
    label: 'Document Analysis',
    icon: FileText,
    color: "text-violet-500",
    bgColor: "bg-violet-100",
    description: "Upload and analyze fund documentation automatically",
    href: "/dashboard/documents"
  },
  {
    label: 'Q&A System',
    icon: HelpCircle,
    color: "text-pink-500",
    bgColor: "bg-pink-100",
    description: "Get instant answers to due diligence queries",
    href: "/dashboard/qa"
  },
  {
    label: 'Risk Assessment',
    icon: AlertTriangle,
    color: "text-orange-500",
    bgColor: "bg-orange-100",
    description: "Monitor real-time risk metrics and alerts",
    href: "/dashboard/risk"
  }
];

export default function DashboardPage() {
  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col gap-y-4 mb-8">
        <h1 className="text-3xl font-bold">Welcome to VAULT Monkeys</h1>
        <p className="text-gray-500">
          Monitor and analyze crypto fund performance and compliance
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => (
          <Card key={stat.label} className="p-6 shadow-lg">
            <div className="flex items-center gap-4">
              <div className={`${stat.bgColor} p-3 rounded-lg`}>
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
              <div>
                <p className="text-gray-500 text-sm">{stat.label}</p>
                <h3 className="text-2xl font-bold">{stat.value}</h3>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {features.map((feature) => (
          <Card 
            key={feature.label} 
            className="p-6 hover:shadow-lg transition-all cursor-pointer bg-white"
          >
            <div className={`${feature.bgColor} w-12 h-12 rounded-lg flex items-center justify-center mb-4`}>
              <feature.icon className={`w-6 h-6 ${feature.color}`} />
            </div>
            <h3 className="text-lg font-semibold mb-2">{feature.label}</h3>
            <p className="text-gray-500 text-sm">{feature.description}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}