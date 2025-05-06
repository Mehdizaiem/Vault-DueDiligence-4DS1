import { UserButton } from "@clerk/nextjs";
import { Bell, Search, Settings } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import logoImage from '../../components/shared/logo.png';
const Navbar = () => {
  return (
    <div className="flex items-center p-4 border-b border-gray-200 justify-between bg-white shadow-sm">
      <div>
        <Link href="/dashboard" className="flex items-center">
          <Image 
            src={logoImage} 
            alt="VAULT Logo" 
            width={50} 
            height={50} 
           
          />
        </Link>
      </div>
      
      <div className="flex items-center gap-6">
        {/* Search Bar */}
        <div className="relative hidden md:flex items-center">
          <Search className="absolute left-3 text-gray-400" size={16} />
          <input 
            type="text" 
            placeholder="Search..." 
            className="bg-gray-100 text-gray-800 border border-gray-200 rounded-full pl-10 pr-4 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          />
        </div>
        
        {/* Notification Bell */}
        <button className="flex items-center justify-center h-9 w-9 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors text-gray-600">
          <Bell size={18} />
        </button>
        
        {/* Settings */}
        <button className="flex items-center justify-center h-9 w-9 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors text-gray-600">
          <Settings size={18} />
        </button>
        
        {/* User Button */}
        <div className="h-9">
          <UserButton 
            afterSignOutUrl="/sign-in" 
            appearance={{
              elements: {
                userButtonAvatarBox: "h-8 w-8"
              }
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default Navbar;