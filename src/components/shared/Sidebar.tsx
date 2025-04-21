"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  FileText, 
  HelpCircle, 
  AlertTriangle,
  TrendingUp, // New icon for Crypto Forecast
  BarChart4, // New icon for Analytics
  Newspaper // New icon for News
} from "lucide-react";

const routes = [
  {
    label: 'Dashboard',
    icon: LayoutDashboard,
    href: '/dashboard',
    color: "text-sky-500",
  },
  {
    label: 'Documents',
    icon: FileText,
    href: '/dashboard/documents',
    color: "text-violet-500",
  },
  {
    label: 'Q&A',
    icon: HelpCircle,
    href: '/dashboard/qa',
    color: "text-pink-500",
  },
  {
    label: 'Risk Analysis',
    icon: AlertTriangle,
    color: "text-orange-500",
    href: '/dashboard/risk',
  },
  {
    label: 'Crypto Forecast',
    icon: TrendingUp,
    color: "text-green-500",
    href: '/dashboard/forecast',
  },
  {
    label: 'Analytics', 
    icon: BarChart4,
    color: "text-indigo-500",
    href: '/dashboard/analytics',
  },
  {
    label: 'News',
    icon: Newspaper,
    color: "text-yellow-500",
    href: '/dashboard/news',
  },
];

const Sidebar = () => {
  const pathname = usePathname();

  return (
    <div className="space-y-4 py-4 flex flex-col h-full bg-[#111827] text-white">
      <div className="px-3 py-2 flex-1">
        <Link href="/dashboard" className="flex items-center pl-3 mb-14">
          <h1 className="text-2xl font-bold">
            VAULT 
          </h1>
        </Link>
        <div className="space-y-1">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className={`text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 rounded-lg transition ${
                pathname === route.href ? 'text-white bg-white/10' : 'text-zinc-400'
              }`}
            >
              <div className="flex items-center flex-1">
                <route.icon className={`h-5 w-5 mr-3 ${route.color}`} />
                {route.label}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;