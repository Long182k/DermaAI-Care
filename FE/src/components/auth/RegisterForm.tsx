import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import { Calendar as CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { AxiosError } from "axios";
import { useAppStore } from "@/store";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ErrorResponseData } from "@/@util/interface/auth.interface";
import { RegisterNewUserParams } from "@/@util/types/auth.type"; // Import the type

interface RegisterFormProps {
  onToggle: () => void;
}

export const RegisterForm = ({ onToggle }: RegisterFormProps) => {
  const { toast } = useToast();
  const [date, setDate] = useState<Date>();

  const { signup } = useAppStore();
  const navigate = useNavigate();

  const [formData, setFormData] = useState<RegisterNewUserParams>({
    userName: "",
    firstName: "",
    lastName: "",
    email: "",
    phoneNumber: "",
    password: "",
    gender: "MALE",
    dateOfBirth: new Date(), // Initialize with a Date object
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      registerMutation.mutate({ ...formData, dateOfBirth: date });

      toast({
        title: "Register successful",
        description: "You have been logged in.",
      });
    } catch (error) {
      toast({
        title: "Register failed",
        description: "Please check your credentials and try again.",
      });
    }
  };

  const registerMutation = useMutation({
    mutationFn: signup,
    onSuccess: (res) => {
      localStorage.setItem("access_token", res.accessToken);
      toast({
        title: "Register successful",
        description: "You have been registered successful.",
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
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold">Create an account</h1>
        <p className="text-muted-foreground">Enter your details to register</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="firstName">First name</Label>
            <Input
              id="firstName"
              placeholder="Enter your first name"
              required
              value={formData.firstName}
              onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="lastName">Last name</Label>
            <Input
              id="lastName"
              placeholder="Enter your last name"
              required
              value={formData.lastName}
              onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            placeholder="Enter your email"
            required
            value={formData.email}
            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="phone">Phone number</Label>
          <Input
            id="phone"
            type="tel"
            placeholder="Enter your phone number"
            required
            value={formData.phoneNumber}
            onChange={(e) => setFormData({ ...formData, phoneNumber: e.target.value })}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              required
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirmPassword">Confirm Password</Label>
            <Input
              id="confirmPassword"
              type="password"
              placeholder="••••••••"
              required
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Gender</Label>
          <Select
            value={formData.gender}
            onValueChange={(value) => setFormData({ ...formData, gender: value })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="MALE">Male</SelectItem>
              <SelectItem value="FEMALE">Female</SelectItem>
              <SelectItem value="OTHER">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Date of birth</Label>
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full justify-start text-left font-normal",
                  !date && "text-muted-foreground"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {date ? format(date, "PPP") : <span>Pick a date</span>}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={date}
                onSelect={setDate}
                initialFocus
              />
            </PopoverContent>
          </Popover>
        </div>

        <Button type="submit" className="w-full">
          Create account
        </Button>

        <div className="text-center mt-6">
          <span className="text-muted-foreground">
            Already have an account?
          </span>{" "}
          <Button type="button" variant="link" onClick={onToggle}>
            Sign in
          </Button>
        </div>
      </form>
    </div>
  );
};
