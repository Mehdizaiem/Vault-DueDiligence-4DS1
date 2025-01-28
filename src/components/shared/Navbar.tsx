import { UserButton } from "@clerk/nextjs";

const Navbar = () => {
  return (
    <div className="flex items-center p-4 border-b justify-between">
      <div>
        <h1 className="font-semibold text-lg">Crypto Due Diligence</h1>
      </div>
      <div className="flex items-center gap-x-3">
        <UserButton afterSignOutUrl="/sign-in" />
      </div>
    </div>
  );
};

export default Navbar;