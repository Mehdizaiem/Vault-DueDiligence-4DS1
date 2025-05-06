// src/components/shared/SidebarWrapper.tsx
'use client';

import { usePathname } from "next/navigation";
import Sidebar from "./Sidebar";

export default function SidebarWrapper() {
  const pathname = usePathname();
  
  // Only hide sidebar on main dashboard page
  const shouldShowSidebar = pathname !== "/dashboard";
  
  if (!shouldShowSidebar) {
    return null;
  }
  
  return (
    <div className="hidden h-full md:flex md:w-72 md:flex-col md:fixed md:inset-y-0 bg-gray-900">
      <Sidebar />
    </div>
  );
}