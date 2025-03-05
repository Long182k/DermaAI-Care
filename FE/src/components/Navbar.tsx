import { Button } from "@/components/ui/button";
import { UserCircle, Globe } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export const Navbar = () => {
  const navigate = useNavigate();

  const languages = [
    { code: "en", name: "English" },
    { code: "fr", name: "French" },
    { code: "ja", name: "Japanese" },
    { code: "Sp", name: "Spanish" },
    { code: "vi", name: "Vietnamese" },
  ];

  const handleLanguageChange = (code: string) => {
    // For now, just log the selection. In a real app, this would update the app's language
    console.log(`Language changed to: ${code}`);
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b">
      <div className="container flex items-center justify-between h-16">
        <Link to="/" className="text-xl font-bold text-primary">
          DermAI Care
        </Link>
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate("/about")}>
            About
          </Button>
          <Button variant="ghost" onClick={() => navigate("/doctors")}>
            Doctors
          </Button>
          <Button variant="ghost" onClick={() => navigate("/contact")}>
            Contact
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <Globe className="h-5 w-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {languages.map((lang) => (
                <DropdownMenuItem
                  key={lang.code}
                  onClick={() => handleLanguageChange(lang.code)}
                >
                  {lang.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          <Button onClick={() => navigate("/auth")}>
            <UserCircle className="mr-2 h-5 w-5" />
            Sign In
          </Button>
        </div>
      </div>
    </nav>
  );
};
