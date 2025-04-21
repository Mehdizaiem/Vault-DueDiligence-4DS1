import { auth } from "@clerk/nextjs";
import { redirect } from "next/navigation";
import Navbar from "@/components/shared/Navbar";
import Sidebar from "@/components/shared/Sidebar";

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
      <div className="hidden h-full md:flex md:w-72 md:flex-col md:fixed md:inset-y-0 bg-gray-900">
        <Sidebar />
      </div>
      <main className="md:pl-72 min-h-screen bg-gray-50">
        <Navbar />
        {children}
      </main>
    </div>
  );
}