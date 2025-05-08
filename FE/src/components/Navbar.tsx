import { Button } from "@/components/ui/button";
import {
  UserCircle,
  Globe,
  Calendar,
  Bell,
  Stethoscope,
  FileImage,
} from "lucide-react";
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

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b">
      <div className="container flex items-center justify-between h-16">
        <Link to="/" className="text-2xl font-bold text-primary tracking-tight">
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
          <Button variant="ghost" onClick={() => navigate("/contact")}>
            Contact
          </Button>
          {/* Appointments Button */}
          {/* <Button variant="ghost" onClick={() => navigate("/appointments")}>
            <Calendar className="h-5 w-5 mr-2" />
            Appointments
          </Button> */}

          {/* Prediction History Button */}
          {/* <Button
            variant="ghost"
            onClick={() => navigate("/prediction-history")}
          >
            <FileImage className="h-5 w-5 mr-2" />
            Prediction History
          </Button> */}

          {/* User Menu */}
          {userInfo && Object.keys(userInfo).length > 0 ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  {userInfo.role === "DOCTOR" ? (
                    <Stethoscope className="mr-2 h-5 w-5" />
                  ) : (
                    <UserCircle className="mr-2 h-5 w-5" />
                  )}
                  Hello, {userInfo.userName}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() =>
                    navigate(
                      userInfo.role === "ADMIN"
                        ? "/admin-dashboard"
                        : "/profile"
                    )
                  }
                >
                  {userInfo.role === "ADMIN" ? "Dashboard" : "Profile"}
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
