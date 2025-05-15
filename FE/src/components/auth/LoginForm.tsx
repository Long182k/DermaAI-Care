import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import { AxiosError } from "axios";
import { forgotPassword } from "@/api/auth";
import { useAppStore } from "@/store";
import { useMutation } from "@tanstack/react-query";
import { Loader2, User } from "lucide-react";
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
  const [forgotEmail, setForgotEmail] = useState("");
  const [showForgotPasswordModal, setShowForgotPasswordModal] = useState(false);

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
      if (res.role === "ADMIN") {
        navigate("/admin-dashboard");
      } else {
        navigate("/");
      }
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

  // Add forgot password mutation
  const forgotPasswordMutation = useMutation({
    mutationFn: forgotPassword,
    onSuccess: () => {
      toast({
        title: "Reset instructions sent",
        description: "Please check your email for password reset instructions.",
      });
      setShowForgotPasswordModal(false);
      setForgotEmail("");
    },
    onError: (error: AxiosError<ErrorResponseData>) => {
      const message =
        error.response?.data?.message || "Failed to send reset instructions";
      toast({
        title: "Reset request failed",
        description: message,
        variant: "destructive",
      });
    },
  });

  const handleForgotPassword = (e: React.FormEvent) => {
    e.preventDefault();
    if (!forgotEmail) {
      toast({
        title: "Email required",
        description: "Please enter your email address",
        variant: "destructive",
      });
      return;
    }
    forgotPasswordMutation.mutate(forgotEmail);
  };

  return (
    <>
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

          <Button
            type="button"
            variant="link"
            className="px-0"
            onClick={() => setShowForgotPasswordModal(true)}
          >
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
            <span className="text-muted-foreground">
              Don't have an account?
            </span>{" "}
            <Button type="button" variant="link" onClick={onToggle}>
              Sign up
            </Button>
          </div>
        </form>
      </div>

      {/* Forgot Password Modal */}
      {showForgotPasswordModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card p-6 rounded-lg shadow-lg w-full max-w-md">
            <h2 className="text-xl font-semibold mb-4">Reset Password</h2>
            <p className="text-muted-foreground mb-4">
              Enter your email address and we'll send you instructions to reset
              your password.
            </p>
            <form onSubmit={handleForgotPassword} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="forgot-email">Email</Label>
                <Input
                  id="forgot-email"
                  type="email"
                  value={forgotEmail}
                  onChange={(e) => setForgotEmail(e.target.value)}
                  placeholder="Enter your email"
                  aria-label="Email for password reset"
                />
              </div>
              <div className="flex justify-end space-x-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setShowForgotPasswordModal(false);
                    setForgotEmail("");
                  }}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={forgotPasswordMutation.isPending}
                >
                  {forgotPasswordMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Sending...
                    </>
                  ) : (
                    "Send Instructions"
                  )}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};
