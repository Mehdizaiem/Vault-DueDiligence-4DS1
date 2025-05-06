// src/components/shared/MainContentWrapper.tsx
'use client';

import { usePathname } from "next/navigation";
import { ReactNode } from "react";

interface MainContentWrapperProps {
  children: ReactNode;
}

export default function MainContentWrapper({ children }: MainContentWrapperProps) {
  const pathname = usePathname();
  
  // Only adjust padding on pages other than main dashboard
  const shouldHavePadding = pathname !== "/dashboard";
  
  return (
    <main className={`min-h-screen bg-gray-50 ${shouldHavePadding ? 'md:pl-72' : ''}`}>
      {children}
    </main>
  );
}