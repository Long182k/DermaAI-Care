import React, { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar as CalendarIcon } from "lucide-react";
import { appointmentApi, Appointment } from "@/api/appointment";
import { format } from "date-fns";

const AppointmentsPage = () => {
  const [activeTab, setActiveTab] = useState("upcoming");
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
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

    fetchAppointments();
  }, []);

  // Filter appointments based on status and date
  const upcomingAppointments = appointments.filter(
    (appointment) =>
      appointment.status === "SCHEDULED" ||
      appointment.status === "CONFIRMED" ||
      appointment.status === "PENDING"
  );

  const pastAppointments = appointments.filter(
    (appointment) =>
      appointment.status === "COMPLETED" || appointment.status === "CANCELLED"
  );

  const handleReschedule = (id: string) => {
    console.log(`Reschedule appointment ${id}`);
  };

  const handleCancel = (id: string) => {
    console.log(`Cancel appointment ${id}`);
  };

  const handlePayment = (id: string) => {
    console.log(`Process payment for appointment ${id}`);
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

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-3xl font-bold mb-6">My Appointments</h1>
        <div className="flex justify-center items-center h-64">
          <p>Loading appointments...</p>
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
    <div className="container mx-auto px-4 py-20">
      <h1 className="text-3xl font-bold mb-6">My Appointments</h1>

      <Tabs defaultValue="upcoming" className="w-full">
        <TabsList className="grid w-full grid-cols-2 mb-8">
          <TabsTrigger
            value="upcoming"
            onClick={() => setActiveTab("upcoming")}
          >
            Upcoming Appointments
          </TabsTrigger>
          <TabsTrigger value="history" onClick={() => setActiveTab("history")}>
            Appointment History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upcoming" className="space-y-4">
          {upcomingAppointments.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center">
                <p className="text-muted-foreground">
                  You have no upcoming appointments.
                </p>
                <Button className="mt-4">Book Appointment</Button>
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
                        variant="outline"
                        size="sm"
                        onClick={() => handleReschedule(appointment.id)}
                      >
                        Reschedule
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCancel(appointment.id)}
                      >
                        Cancel
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
                  You have no appointment history.
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
    </div>
  );
};

export default AppointmentsPage;
