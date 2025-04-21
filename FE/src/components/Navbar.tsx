import { Button } from "@/components/ui/button";
import { UserCircle, Globe, Calendar, Bell } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAppStore } from "@/store";
import { useMutation } from "@tanstack/react-query";

import { AxiosError } from "axios";
import { ErrorResponseData } from "@/@util/interface/auth.interface";
import { useToast } from "@/hooks/use-toast";

export const Navbar = () => {
  const { userInfo, logout } = useAppStore();
  const { toast } = useToast();

  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      logoutMutation.mutate();

      toast({
        title: "Login successful",
        description: "You have been logged in.",
      });
    } catch (error) {
      toast({
        title: "Login failed",
        description: "Please check your credentials and try again.",
      });
    }
  };

  const logoutMutation = useMutation({
    mutationFn: logout,
    onSuccess: () => {
      toast({
        title: "Logout successful",
        description: "You have been logout.",
      });
      navigate("/auth");
    },
    onError: (error: AxiosError<ErrorResponseData>) => {
      if (error.response?.status === 401) {
        const message = error.response?.data?.message;
        toast({
          title: "Login failed",
          description: message,
        });
      } else {
        toast({
          title: "Login failed",
          description: "Try Again",
        });
      }
    },
  });

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

  // Mock notification count for demo purposes
  const notificationCount = 3;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b">
      <div className="container flex items-center justify-between h-16">
        <Link to="/" className="text-xl font-bold text-primary">
          DermAI Care
        </Link>

        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate("/")}>
            Home
          </Button>
          <Button variant="ghost" onClick={() => navigate("/about")}>
            About
          </Button>
          <Button variant="ghost" onClick={() => navigate("/services")}>
            Services
          </Button>
          <Button variant="ghost" onClick={() => navigate("/doctors")}>
            Doctors
          </Button>
          <Button variant="ghost" onClick={() => navigate("/contact")}>
            Contact
          </Button>

          {/* Appointments Button */}
          <Button variant="ghost" onClick={() => navigate("/appointments")}>
            <Calendar className="h-5 w-5 mr-2" />
            Appointments
          </Button>

          {/* Notifications Button */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-5 w-5" />
                {notificationCount > 0 && (
                  <Badge className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 bg-red-500">
                    {notificationCount}
                  </Badge>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80">
              <DropdownMenuItem className="font-bold border-b p-3">
                Notifications
              </DropdownMenuItem>
              <DropdownMenuItem>
                <div className="flex flex-col py-2">
                  <span className="font-semibold">Appointment Confirmed</span>
                  <span className="text-sm text-muted-foreground">
                    Your appointment with Dr. Smith is confirmed for tomorrow at
                    2:00 PM
                  </span>
                </div>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <div className="flex flex-col py-2">
                  <span className="font-semibold">Payment Successful</span>
                  <span className="text-sm text-muted-foreground">
                    Your payment of $150 for the last appointment was successful
                  </span>
                </div>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <div className="flex flex-col py-2">
                  <span className="font-semibold">Prescription Ready</span>
                  <span className="text-sm text-muted-foreground">
                    Your prescription is ready to be picked up
                  </span>
                </div>
              </DropdownMenuItem>
              <DropdownMenuItem className="justify-center text-primary">
                View all notifications
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Language Selector */}
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

          {/* User Menu */}
          {userInfo && Object.keys(userInfo).length > 0 ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  <UserCircle className="mr-2 h-5 w-5" />
                  Hello, {userInfo.userName}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => navigate("/profile")}>
                  Profile
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => {
                    handleLogout();
                    navigate("/");
                  }}
                >
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button onClick={() => navigate("/auth")}>
              <UserCircle className="mr-2 h-5 w-5" />
              Sign In
            </Button>
          )}
        </div>
      </div>
    </nav>
  );
};
