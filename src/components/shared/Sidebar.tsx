"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { 
  LayoutDashboard, 
  FileText, 
  HelpCircle, 
  AlertTriangle,
  TrendingUp,
  BarChart4,
  Newspaper,
  ChevronRight
} from "lucide-react";

const routes = [
  {
    label: 'Dashboard',
    icon: LayoutDashboard,
    href: '/dashboard',
    color: "text-blue-400",
    lightColor: '#4c6bff',
  },
  {
    label: 'Documents',
    icon: FileText,
    href: '/dashboard/documents',
    color: "text-violet-400",
    lightColor: '#8b5cf6',
  },
  {
    label: 'Q&A',
    icon: HelpCircle,
    href: '/dashboard/qa',
    color: "text-pink-400",
    lightColor: '#ec4899',
  },
  {
    label: 'Risk Analysis',
    icon: AlertTriangle,
    color: "text-orange-400",
    href: '/dashboard/risk',
    lightColor: '#f97316',
  },
  // New Routes
  {
    label: 'Crypto Forecast',
    icon: TrendingUp,
    color: "text-green-400",
    href: '/dashboard/forecast',
    lightColor: '#10b981',
  },
  {
    label: 'Analytics',
    icon: BarChart4,
    color: "text-indigo-400",
    href: '/dashboard/analytics',
    lightColor: '#6366f1',
  },
  {
    label: 'News',
    icon: Newspaper,
    color: "text-yellow-400",
    href: '/dashboard/news',
    lightColor: '#eab308',
  },
   {
    label: 'Report Generation',
    icon: FileText,
    color: "text-green-400",
    href: '/dashboard/reports',
    lightColor: '#BBBE33FF',
  }
];

const Sidebar = () => {
  const pathname = usePathname();
  const [hoveredItem, setHoveredItem] = useState<number | null>(null);

  return (
    <div className="space-y-4 py-4 flex flex-col h-full relative">
      {/* Logo Section with Glow Effect */}
      <div className="px-3 py-2 flex-1">
        <Link 
          href="/dashboard" 
          className="flex items-center pl-3 mb-14"
        >
          <h1 
            className="text-2xl font-bold text-white"
            style={{
              textShadow: '0 0 10px rgba(76, 107, 255, 0.8), 0 0 20px rgba(76, 107, 255, 0.4)'
            }}
          >
            VAULT
          </h1>
        </Link>
        
        {/* Navigation Links */}
        <div className="space-y-2">
          {routes.map((route, index) => {
            const isActive = pathname === route.href;
            
            return (
              <Link
                key={route.href}
                href={route.href}
                className={`relative group flex p-3 w-full justify-start font-medium cursor-pointer rounded-lg transition-all duration-300 ease-in-out ${
                  isActive 
                    ? 'text-white bg-[#1a1a3a]' 
                    : 'text-gray-400 hover:text-white hover:bg-[#1a1a3a]/50'
                }`}
                onMouseEnter={() => setHoveredItem(index)}
                onMouseLeave={() => setHoveredItem(null)}
                style={{
                  background: isActive 
                    ? `linear-gradient(90deg, rgba(26, 26, 58, 0.8), rgba(26, 26, 58, 0.4))` 
                    : hoveredItem === index 
                      ? `linear-gradient(90deg, rgba(26, 26, 58, 0.4), rgba(26, 26, 58, 0.1))` 
                      : '',
                  borderLeft: isActive 
                    ? `2px solid ${route.lightColor}` 
                    : '2px solid transparent'
                }}
              >
                <div className="flex items-center flex-1 text-sm">
                  <route.icon className={`h-5 w-5 mr-3 ${route.color}`} />
                  {route.label}
                  
                  {isActive && (
                    <ChevronRight 
                      size={16} 
                      className="ml-auto text-white" 
                    />
                  )}
                </div>
                
                {/* Glow effect */}
                {(isActive || hoveredItem === index) && (
                  <div 
                    className="absolute inset-0 rounded-lg opacity-20 pointer-events-none"
                    style={{
                      background: `radial-gradient(circle at left, ${route.lightColor}40 0%, transparent 70%)`,
                    }}
                  />
                )}
                
                {/* Animated underline */}
                <div 
                  className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r transition-all duration-300 ease-out"
                  style={{
                    width: isActive ? '100%' : hoveredItem === index ? '50%' : '0%',
                    background: `linear-gradient(to right, ${route.lightColor}, transparent)`,
                  }}
                />
              </Link>
            );
          })}
        </div>
      </div>
      
      {/* Bottom decoration */}
      <div className="px-3 py-4">
        <div className="h-20 rounded-xl bg-[#1a1a3a]/30 relative overflow-hidden">
          <div 
            className="absolute inset-0 opacity-30"
            style={{
              background: 'radial-gradient(circle at center, #4c6bff 0%, transparent 70%)',
              animation: 'pulse 4s infinite ease-in-out'
            }}
          />
          
          <div className="p-4 relative z-10 flex flex-col justify-center h-full">
            <div className="text-xs text-blue-300 mb-1">Version</div>
            <div className="text-sm text-white font-mono">VAULT 1.0.0</div>
          </div>
        </div>
      </div>
      
      {/* Background animated gradients */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        <div 
          className="absolute top-0 left-0 w-full h-40 opacity-20"
          style={{
            background: 'radial-gradient(circle at top, #4c6bff 0%, transparent 70%)',
          }}
        />
        
        <div 
          className="absolute bottom-0 left-0 w-full h-40 opacity-20"
          style={{
            background: 'radial-gradient(circle at bottom, #4c6bff 0%, transparent 70%)',
          }}
        />
      </div>
      
      {/* CSS Animations */}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 0.3;
            transform: scale(1);
          }
          50% {
            opacity: 0.5;
            transform: scale(1.1);
          }
        }
      `}</style>
    </div>
  );
};

export default Sidebar;