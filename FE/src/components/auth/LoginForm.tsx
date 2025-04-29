import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import { AxiosError } from "axios";

import { useAppStore } from "@/store";
import { useMutation } from "@tanstack/react-query";
import { User } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ErrorResponseData } from "@/@util/interface/auth.interface";

interface LoginFormProps {
  onToggle: () => void;
}

export const LoginForm = ({ onToggle }: LoginFormProps) => {
  const { toast } = useToast();
  const { login } = useAppStore();
  const navigate = useNavigate();

  const [formData, setFormData] = useState({ username: "", password: "" });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      loginMutation.mutate(formData);

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

  const loginMutation = useMutation({
    mutationFn: login,
    onSuccess: (res) => {
      toast({
        title: "Login successful",
        description: "You have been logged in.",
      });
      navigate("/");

      // Navigate based on role
      // if (res.role === "ADMIN") {
      //   navigate("/dashboard");
      // } else {
      //   navigate("/");
      // }
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
    <div className="bg-card p-8 rounded-lg shadow-lg">
      <div className="flex flex-col items-center gap-6 mb-8">
        <div className="h-12 w-12 bg-primary/10 rounded-full flex items-center justify-center">
          <User className="h-6 w-6 text-primary" />
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-bold">Welcome back</h1>
          <p className="text-muted-foreground">Sign in to your account</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            placeholder="Enter your email"
            required
            value={formData.username}
            onChange={(e) =>
              setFormData({ ...formData, username: e.target.value })
            }
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            placeholder="••••••••"
            required
            value={formData.password}
            onChange={(e) =>
              setFormData({ ...formData, password: e.target.value })
            }
          />
        </div>

        <Button type="button" variant="link" className="px-0">
          Forgot password?
        </Button>

        <Button type="submit" className="w-full">
          Sign in
        </Button>

        <div className="relative my-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t"></div>
          </div>
        </div>

        <div className="text-center mt-6">
          <span className="text-muted-foreground">Don't have an account?</span>{" "}
          <Button type="button" variant="link" onClick={onToggle}>
            Sign up
          </Button>
        </div>
      </form>
    </div>
  );
};
