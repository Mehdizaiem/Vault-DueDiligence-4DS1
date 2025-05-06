// src/app/(dashboard)/layout.tsx
import { auth } from "@clerk/nextjs";
import { redirect } from "next/navigation";
import Navbar from "@/components/shared/Navbar";
import SidebarWrapper from "@/components/shared/SidebarWrapper";
import MainContentWrapper from "@/components/shared/MainContentWrapper";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { userId } = auth();

  if (!userId) {
    redirect("/sign-in");
  }

  return (
    <div className="h-full relative bg-gray-50">
      <SidebarWrapper />
      <MainContentWrapper>
        <Navbar />
        {children}
      </MainContentWrapper>
    </div>
  );
}