import { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: ReactNode;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  iconBgColor?: string;
  className?: string;
}

export default function MetricCard({ 
  title, 
  value, 
  icon, 
  subtitle, 
  trend, 
  iconBgColor = 'bg-blue-100',
  className
}: MetricCardProps) {
  return (
    <Card className={cn("hover:shadow-md transition-shadow duration-200", className)}>
      <CardContent className="pt-6">
        <div className="flex items-center space-x-3">
          <div className={`${iconBgColor} p-3 rounded-lg`}>
            {icon}
          </div>
          <h3 className="text-sm font-medium text-gray-700">{title}</h3>
        </div>

        <div className="mt-4">
          <div className="text-2xl font-bold">
            {value}
          </div>
          {subtitle && (
            <div className={`text-xs mt-1 ${
              trend === 'up' ? 'text-green-500' : 
              trend === 'down' ? 'text-red-500' : 
              'text-gray-500'
            }`}>
              {subtitle}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}