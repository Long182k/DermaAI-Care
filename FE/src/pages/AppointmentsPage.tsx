import { Appointment, appointmentApi } from "@/api/appointment";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { useAppStore } from "@/store";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  format,
  isFuture,
  isThisMonth,
  isThisWeek,
  isToday
} from "date-fns";
import { Calendar as CalendarIcon, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

// Define filter types
type FilterType = "day" | "week" | "month" | "all";

const AppointmentsPage = () => {
  const [activeTab, setActiveTab] = useState("upcoming");
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterType>("all");
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { userInfo } = useAppStore();
  const isDoctor = userInfo?.role === "DOCTOR";

  // Set default active tab based on role
  useEffect(() => {
    if (isDoctor) {
      setActiveTab("doctor-appointments");
    }
  }, [isDoctor]);

  const navigate = useNavigate();

  // Add useMutation hook for canceling appointments
  const cancelAppointmentMutation = useMutation({
    mutationFn: (id: string) => appointmentApi.cancelAppointment(id),
    onSuccess: () => {
      toast({
        title: "Appointment cancelled",
        description: "Your appointment has been successfully cancelled.",
        variant: "default",
      });
      // Refetch appointments after cancellation
      fetchAppointments();
    },
    onError: (error) => {
      console.error("Error cancelling appointment:", error);
      toast({
        title: "Failed to cancel appointment",
        description: "Please try again later.",
        variant: "destructive",
      });
    },
  });

  const fetchAppointments = async () => {
    try {
      setLoading(true);
      const data = await appointmentApi.getUserAppointments();
      setAppointments(data);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch appointments:", err);
      setError("Failed to load appointments. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAppointments();
  }, []);

  // Filter appointments based on status and date
  const filterAppointmentsByDate = (appointments: Appointment[]) => {
    return appointments.filter((appointment) => {
      const appointmentDate = new Date(appointment.Schedule.startTime);
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      // Only include future appointments
      if (!isFuture(appointmentDate) && !isToday(appointmentDate)) {
        return false;
      }

      switch (filter) {
        case "day":
          return isToday(appointmentDate);
        case "week":
          return isThisWeek(appointmentDate);
        case "month":
          return isThisMonth(appointmentDate);
        case "all":
        default:
          return true;
      }
    });
  };

  // Filter appointments based on status
  const upcomingAppointments = filterAppointmentsByDate(
    appointments.filter(
      (appointment) =>
        appointment.status === "SCHEDULED" || appointment.status === "PENDING"
    )
  );

  const pastAppointments = filterAppointmentsByDate(
    appointments.filter(
      (appointment) =>
        appointment.status === "CONFIRMED" || appointment.status === "CANCELLED"
    )
  );

  // For doctor view - all upcoming appointments with patients
  const doctorUpcomingAppointments = appointments.filter((appointment) => {
    const appointmentDate = new Date(appointment.Schedule.startTime);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    return (
      (appointment.status === "SCHEDULED" ||
        appointment.status === "PENDING" ||
        appointment.status === "CONFIRMED") &&
      (isFuture(appointmentDate) || isToday(appointmentDate))
    );
  });

  const handleCancel = (id: string) => {
    // Show loading state and disable button during cancellation
    cancelAppointmentMutation.mutate(id);
  };

  const handlePayment = (id: string) => {
    // Navigate to payment page
    window.location.href = `/payment/${id}`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return format(date, "EEEE, MMMM d, yyyy");
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return format(date, "h:mm a");
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "CONFIRMED":
        return "bg-green-100 text-green-800";
      case "SCHEDULED":
      case "PENDING":
        return "bg-yellow-100 text-yellow-800";
      case "COMPLETED":
        return "bg-blue-100 text-blue-800";
      case "CANCELLED":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getDoctorName = (appointment: Appointment) => {
    const { Doctor } = appointment;
    if (Doctor.firstName && Doctor.lastName) {
      return `Dr. ${Doctor.firstName} ${Doctor.lastName}`;
    } else if (Doctor.firstName) {
      return `Dr. ${Doctor.firstName}`;
    } else if (Doctor.lastName) {
      return `Dr. ${Doctor.lastName}`;
    } else {
      return Doctor.userName;
    }
  };

  const getPatientName = (appointment: Appointment) => {
    const { Patient } = appointment;
    if (Patient.firstName && Patient.lastName) {
      return `${Patient.firstName} ${Patient.lastName}`;
    } else if (Patient.firstName) {
      return Patient.firstName;
    } else if (Patient.lastName) {
      return Patient.lastName;
    } else {
      return Patient.userName;
    }
  };

  const getFilterLabel = () => {
    switch (filter) {
      case "day":
        return "Day View";
      case "week":
        return "Week View";
      case "month":
        return "Month View";
      case "all":
      default:
        return "All Schedules";
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-3xl font-bold mb-6">My Appointments</h1>
        <div className="flex justify-center items-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-3xl font-bold mb-6">My Appointments</h1>
        <div className="flex justify-center items-center h-64">
          <p className="text-red-500">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Navbar />
      <div className="container mx-auto px-4 py-20">
        <div className="flex items-center mb-6">
          <Button
            variant="ghost"
            size="icon"
            className="mr-2"
            onClick={() => navigate("/services")}
            aria-label="Back"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </Button>
          <h1 className="text-3xl font-bold">
            {isDoctor ? "Appointments Management" : "My Appointments"}
          </h1>
        </div>

        {isDoctor ? (
          <div className="mb-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-medium">All Patient Appointments</h2>

              {/* Filter dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="w-[120px]">
                    {getFilterLabel()}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => setFilter("day")}>
                    Day View
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setFilter("week")}>
                    Week View
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setFilter("month")}>
                    Month View
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setFilter("all")}>
                    All Schedules
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        ) : (
          // Patient view - show regular tabs
          <div className="mb-6">
            <Tabs value={activeTab} className="w-full">
              <div className="flex justify-between items-center">
                <TabsList className="w-[calc(100%-120px)] grid grid-cols-2">
                  <TabsTrigger
                    value="upcoming"
                    onClick={() => setActiveTab("upcoming")}
                  >
                    Upcoming Appointments
                  </TabsTrigger>
                  <TabsTrigger
                    value="history"
                    onClick={() => setActiveTab("history")}
                  >
                    Appointment History
                  </TabsTrigger>
                </TabsList>

                {/* Filter dropdown */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="w-[120px]">
                      {getFilterLabel()}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => setFilter("day")}>
                      Day View
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setFilter("week")}>
                      Week View
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setFilter("month")}>
                      Month View
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setFilter("all")}>
                      All Schedules
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </Tabs>
          </div>
        )}

        {isDoctor ? (
          // Doctor view - only show patient appointments content
          <div className="space-y-4">
            {doctorUpcomingAppointments.length === 0 ? (
              <Card>
                <CardContent className="py-10 text-center">
                  <p className="text-muted-foreground">
                    You have no upcoming appointments with patients
                    {filter !== "all" ? ` for this ${filter}` : ""}.
                  </p>
                </CardContent>
              </Card>
            ) : (
              doctorUpcomingAppointments.map((appointment) => (
                <Card key={appointment.id} className="overflow-hidden">
                  <div className="flex flex-col md:flex-row">
                    <div className="bg-primary/10 p-6 flex flex-col justify-center items-center md:w-1/4">
                      <CalendarIcon className="h-8 w-8 text-primary mb-2" />
                      <p className="text-lg font-medium">
                        {formatDate(appointment.Schedule.startTime)}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {formatTime(appointment.Schedule.startTime)}
                      </p>
                    </div>

                    <CardContent className="flex-1 p-6">
                      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                        <div>
                          <h3 className="text-xl font-semibold">
                            {getPatientName(appointment)}
                          </h3>
                          <p className="text-muted-foreground">
                            Gender: {appointment.Patient.gender}
                          </p>
                          {appointment.Patient.phoneNumber && (
                            <p className="text-muted-foreground">
                              Phone: {appointment.Patient.phoneNumber}
                            </p>
                          )}
                          {appointment.Patient.email && (
                            <p className="text-muted-foreground">
                              Email: {appointment.Patient.email}
                            </p>
                          )}
                        </div>
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium mt-2 md:mt-0 ${getStatusColor(
                            appointment.status
                          )}`}
                        >
                          {appointment.status}
                        </span>
                      </div>

                      {appointment.notes && (
                        <div className="mt-2 mb-4">
                          <p className="font-medium">Patient Notes:</p>
                          <p className="text-muted-foreground">
                            {appointment.notes}
                          </p>
                        </div>
                      )}

                      <div className="flex flex-wrap gap-2 mt-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            // Navigate to prediction history with patient ID
                            navigate(
                              `/prediction-history/${appointment.Patient.id}`
                            );
                          }}
                        >
                          View Patient Details
                        </Button>
                      </div>
                    </CardContent>
                  </div>
                </Card>
              ))
            )}
          </div>
        ) : (
          // Patient view - show regular tabs content
          <Tabs value={activeTab} className="w-full">
            <TabsContent value="upcoming" className="space-y-4">
              {upcomingAppointments.length === 0 ? (
                <Card>
                  <CardContent className="py-10 text-center">
                    <p className="text-muted-foreground">
                      You have no upcoming appointments
                      {filter !== "all" ? ` for this ${filter}` : ""}.
                    </p>
                    <Button
                      className="mt-4"
                      onClick={() => navigate("/doctors")}
                    >
                      Book Appointment
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                upcomingAppointments.map((appointment) => (
                  <Card key={appointment.id} className="overflow-hidden">
                    <div className="flex flex-col md:flex-row">
                      <div className="bg-primary/10 p-6 flex flex-col justify-center items-center md:w-1/4">
                        <CalendarIcon className="h-8 w-8 text-primary mb-2" />
                        <p className="text-lg font-medium">
                          {formatDate(appointment.Schedule.startTime)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {formatTime(appointment.Schedule.startTime)}
                        </p>
                      </div>

                      <CardContent className="flex-1 p-6">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                          <div>
                            <h3 className="text-xl font-semibold">
                              {getDoctorName(appointment)}
                            </h3>
                            <p className="text-muted-foreground">
                              {appointment.Doctor.education || "Specialist"}
                            </p>
                          </div>
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-medium mt-2 md:mt-0 ${getStatusColor(
                              appointment.status
                            )}`}
                          >
                            {appointment.status}
                          </span>
                        </div>

                        <div className="flex flex-wrap gap-2 mt-4">
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleCancel(appointment.id)}
                            disabled={
                              cancelAppointmentMutation.isPending &&
                              cancelAppointmentMutation.variables ===
                                appointment.id
                            }
                          >
                            {cancelAppointmentMutation.isPending &&
                            cancelAppointmentMutation.variables ===
                              appointment.id ? (
                              <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Cancelling...
                              </>
                            ) : (
                              "Cancel"
                            )}
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => handlePayment(appointment.id)}
                          >
                            Pay Now
                          </Button>
                        </div>
                      </CardContent>
                    </div>
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="history" className="space-y-4">
              {pastAppointments.length === 0 ? (
                <Card>
                  <CardContent className="py-10 text-center">
                    <p className="text-muted-foreground">
                      You have no appointment history
                      {filter !== "all" ? ` for this ${filter}` : ""}.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                pastAppointments.map((appointment) => (
                  <Card key={appointment.id} className="overflow-hidden">
                    <div className="flex flex-col md:flex-row">
                      <div className="bg-secondary/10 p-6 flex flex-col justify-center items-center md:w-1/4">
                        <CalendarIcon className="h-8 w-8 text-secondary mb-2" />
                        <p className="text-lg font-medium">
                          {formatDate(appointment.Schedule.startTime)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {formatTime(appointment.Schedule.startTime)}
                        </p>
                      </div>

                      <CardContent className="flex-1 p-6">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                          <div>
                            <h3 className="text-xl font-semibold">
                              {getDoctorName(appointment)}
                            </h3>
                            <p className="text-muted-foreground">
                              {appointment.Doctor.education || "Specialist"}
                            </p>
                          </div>
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-medium mt-2 md:mt-0 ${getStatusColor(
                              appointment.status
                            )}`}
                          >
                            {appointment.status}
                          </span>
                        </div>

                        {appointment.status === "COMPLETED" && (
                          <div className="flex flex-wrap gap-2 mt-4">
                            <Button variant="outline" size="sm">
                              View Details
                            </Button>
                            <Button variant="outline" size="sm">
                              Book Again
                            </Button>
                          </div>
                        )}
                      </CardContent>
                    </div>
                  </Card>
                ))
              )}
            </TabsContent>
          </Tabs>
        )}
      </div>
    </>
  );
};

export default AppointmentsPage;
