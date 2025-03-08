import React, { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Calendar,
  Clock,
  Calendar as CalendarIcon,
  ChevronRight,
} from "lucide-react";

// Mock appointments data
const upcomingAppointments = [
  {
    id: 1,
    doctor: "Dr. Sarah Johnson",
    specialty: "Cardiology",
    date: "2023-12-15",
    time: "10:00 AM",
    status: "Confirmed",
  },
  {
    id: 2,
    doctor: "Dr. Michael Chen",
    specialty: "Dermatology",
    date: "2023-12-20",
    time: "2:30 PM",
    status: "Pending",
  },
];

const pastAppointments = [
  {
    id: 3,
    doctor: "Dr. Emily Wilson",
    specialty: "Neurology",
    date: "2023-11-05",
    time: "9:15 AM",
    status: "Completed",
  },
  {
    id: 4,
    doctor: "Dr. James Miller",
    specialty: "Orthopedics",
    date: "2023-10-22",
    time: "11:45 AM",
    status: "Completed",
  },
  {
    id: 5,
    doctor: "Dr. Patricia Garcia",
    specialty: "Pediatrics",
    date: "2023-09-18",
    time: "3:00 PM",
    status: "Cancelled",
  },
];

const AppointmentsPage = () => {
  const [activeTab, setActiveTab] = useState("upcoming");

  const handleReschedule = (id: number) => {
    console.log(`Reschedule appointment ${id}`);
  };

  const handleCancel = (id: number) => {
    console.log(`Cancel appointment ${id}`);
  };

  const handlePayment = (id: number) => {
    console.log(`Process payment for appointment ${id}`);
    // Navigate to payment page
    window.location.href = `/payment/${id}`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Confirmed":
        return "bg-green-100 text-green-800";
      case "Pending":
        return "bg-yellow-100 text-yellow-800";
      case "Completed":
        return "bg-blue-100 text-blue-800";
      case "Cancelled":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

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
                      {formatDate(appointment.date)}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {appointment.time}
                    </p>
                  </div>

                  <CardContent className="flex-1 p-6">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                      <div>
                        <h3 className="text-xl font-semibold">
                          {appointment.doctor}
                        </h3>
                        <p className="text-muted-foreground">
                          {appointment.specialty}
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
                      {formatDate(appointment.date)}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {appointment.time}
                    </p>
                  </div>

                  <CardContent className="flex-1 p-6">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                      <div>
                        <h3 className="text-xl font-semibold">
                          {appointment.doctor}
                        </h3>
                        <p className="text-muted-foreground">
                          {appointment.specialty}
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

                    {appointment.status === "Completed" && (
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
