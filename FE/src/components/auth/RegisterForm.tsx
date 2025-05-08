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
import { RegisterNewUserParams } from "@/@util/types/auth.type";

interface RegisterFormProps {
  onToggle: () => void;
}

export const RegisterForm = ({ onToggle }: RegisterFormProps) => {
  const { toast } = useToast();
  const [date, setDate] = useState<Date>();
  const [calendarViewDate, setCalendarViewDate] = useState<Date>(new Date());
  const [confirmPassword, setConfirmPassword] = useState<string>("");

  // Handler for when the calendar's month or year changes
  const handleMonthYearChange = (newMonth: Date) => {
    setCalendarViewDate(newMonth);
    // If no date is selected or the selected date is not in the new month/year, update to first day of month
    if (
      !date ||
      date.getMonth() !== newMonth.getMonth() ||
      date.getFullYear() !== newMonth.getFullYear()
    ) {
      setDate(new Date(newMonth.getFullYear(), newMonth.getMonth(), 1));
    }
  };

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
    dateOfBirth: new Date(),
    role: "PATIENT",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Check if passwords match
    if (formData.password !== confirmPassword) {
      toast({
        title: "Password mismatch",
        description: "Password and confirm password do not match.",
        variant: "destructive",
      });
      return; // Prevent form submission
    }

    try {
      registerMutation.mutate({
        ...formData,
        dateOfBirth: date,
        userName: formData.lastName,
        role: "PATIENT",
      });

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

      // Use window.location.href to navigate and refresh the page
      window.location.href = "/auth";
    },
    onError: (error: AxiosError<ErrorResponseData>) => {
      const { statusCode, message } = error.response.data;
      if (statusCode === 400) {
        toast({
          title: "Register failed",
          description: message,
        });
      } else {
        toast({
          title: "Register failed",
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
              onChange={(e) =>
                setFormData({ ...formData, firstName: e.target.value })
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="lastName">Last name</Label>
            <Input
              id="lastName"
              placeholder="Enter your last name"
              required
              value={formData.lastName}
              onChange={(e) =>
                setFormData({ ...formData, lastName: e.target.value })
              }
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
            onChange={(e) =>
              setFormData({ ...formData, email: e.target.value })
            }
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
            onChange={(e) =>
              setFormData({ ...formData, phoneNumber: e.target.value })
            }
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
              onChange={(e) =>
                setFormData({ ...formData, password: e.target.value })
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirmPassword">Confirm Password</Label>
            <Input
              id="confirmPassword"
              type="password"
              placeholder="••••••••"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label>Gender</Label>
          <Select
            value={formData.gender}
            onValueChange={(value) =>
              setFormData({ ...formData, gender: value })
            }
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
          <div className="grid grid-cols-3 gap-2">
            <Select
              value={date ? date.getDate().toString() : ""}
              onValueChange={(value) => {
                const newDate = new Date(
                  date ? date.getTime() : new Date().getTime()
                );
                newDate.setDate(parseInt(value));
                setDate(newDate);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Day" />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: 31 }, (_, i) => i + 1).map((day) => (
                  <SelectItem key={day} value={day.toString()}>
                    {day}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={date ? date.getMonth().toString() : ""}
              onValueChange={(value) => {
                const newDate = new Date(
                  date ? date.getTime() : new Date().getTime()
                );
                newDate.setMonth(parseInt(value));
                setDate(newDate);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Month" />
              </SelectTrigger>
              <SelectContent>
                {[
                  "January",
                  "February",
                  "March",
                  "April",
                  "May",
                  "June",
                  "July",
                  "August",
                  "September",
                  "October",
                  "November",
                  "December",
                ].map((month, index) => (
                  <SelectItem key={month} value={index.toString()}>
                    {month}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={date ? date.getFullYear().toString() : ""}
              onValueChange={(value) => {
                const newDate = new Date(
                  date ? date.getTime() : new Date().getTime()
                );
                newDate.setFullYear(parseInt(value));
                setDate(newDate);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Year" />
              </SelectTrigger>
              <SelectContent className="max-h-[200px] overflow-y-auto">
                {Array.from(
                  { length: new Date().getFullYear() - 1900 + 1 },
                  (_, i) => new Date().getFullYear() - i
                ).map((year) => (
                  <SelectItem key={year} value={year.toString()}>
                    {year}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {date && (
            <p className="text-sm text-muted-foreground mt-1">
              {format(date, "PPP")}
            </p>
          )}
        </div>

        <div className="space-y-2"></div>

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
