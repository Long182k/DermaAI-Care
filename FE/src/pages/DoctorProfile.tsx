import { Navbar } from "@/components/Navbar";
import { useParams, useNavigate } from "react-router-dom";
import { Calendar, ArrowLeft, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  doctorApi,
  Schedule,
  ScheduleFilter,
  appointmentApi,
  CreateAppointmentDto,
} from "@/api/appointment";
import { Button } from "@/components/ui/button";
import { format, parseISO, startOfWeek, addDays, addWeeks } from "date-fns";
import { useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useAppStore } from "@/store";

const DoctorProfile = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [currentWeek, setCurrentWeek] = useState(0);
  const [filter, setFilter] = useState<ScheduleFilter>("week");
  const [selectedSchedule, setSelectedSchedule] = useState<Schedule | null>(
    null
  );
  const [notes, setNotes] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { userInfo } = useAppStore();
  // Get current date range for schedules
  const today = new Date();
  const startDate = addWeeks(startOfWeek(today), currentWeek);
  const endDate = addDays(startDate, 6);

  // Function to handle schedule selection and open modal
  const handleScheduleSelect = (schedule: Schedule) => {
    setSelectedSchedule(schedule);
    setNotes("");
    setIsModalOpen(true);
    // Add schedule ID to URL without navigating
    window.history.pushState(
      {},
      "",
      `${window.location.pathname}?scheduleId=${schedule.id}`
    );
  };

  // Function to handle booking appointment after notes are added
  const handleBookAppointment = () => {
    if (selectedSchedule) {
      const userId = userInfo?.userId ?? userInfo?.id;

      // Clear schedule ID from URL before navigating
      window.history.pushState({}, "", window.location.pathname);
      createAppointmentMutation.mutate({
        patientId: userId,
        scheduleId: selectedSchedule.id,
        notes: notes,
      });
    }

    setIsModalOpen(false);
  };

  // Add useMutation hook for appointment creation
  const createAppointmentMutation = useMutation({
    mutationFn: (data: CreateAppointmentDto) =>
      appointmentApi.createAppointment(data),
    onSuccess: () => {
      toast({
        title: "Appointment booked successfully",
        description: "Your appointment has been scheduled.",
        variant: "default",
      });
      navigate("/appointments");
    },
    onError: (error) => {
      const { response } = error;
      toast({
        title: "Failed to book appointment",
        description:
          response.data.message !== undefined
            ? response.data.message
            : "Please try again later.",
        variant: "destructive",
      });
      console.error("Appointment booking error:", error);
    },
  });

  // Function to handle modal close
  const handleModalClose = (open: boolean) => {
    if (!open) {
      // Clear schedule ID from URL when modal is closed
      window.history.pushState({}, "", window.location.pathname);
    }
    setIsModalOpen(open);
  };

  const {
    data: doctor,
    isLoading: isLoadingDoctor,
    error: doctorError,
  } = useQuery({
    queryKey: ["doctor", id],
    queryFn: () => doctorApi.getDoctorById(id!),
    enabled: !!id,
  });

  const {
    data: schedules,
    isLoading: isLoadingSchedules,
    error: schedulesError,
  } = useQuery({
    queryKey: ["doctorSchedules", id, startDate, endDate, filter],
    queryFn: () =>
      doctorApi.getDoctorSchedules(id!, startDate, endDate, filter),
    enabled: !!id,
  });

  // Group schedules by day
  const schedulesByDay = schedules?.reduce(
    (acc: Record<string, Schedule[]>, schedule: Schedule) => {
      const day = format(parseISO(schedule.startTime), "EEEE"); // Get day name (Monday, Tuesday, etc.)
      if (!acc[day]) {
        acc[day] = [];
      }
      acc[day].push(schedule);
      return acc;
    },
    {}
  );

  const handleFilterChange = (newFilter: ScheduleFilter) => {
    setFilter(newFilter);
    // Reset to current week when changing filters
    setCurrentWeek(0);
  };

  const getFilterLabel = () => {
    switch (filter) {
      case "day":
        return "Day View";
      case "week":
        return "Week View";
      case "month":
        return "Month View";
      default:
        return "All Schedules";
    }
  };

  if (isLoadingDoctor || isLoadingSchedules) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="pt-20">
          <div className="container py-12">
            <div className="flex justify-center items-center py-20">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (doctorError || !doctor) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="pt-20">
          <div className="container py-12">
            <div className="text-center text-red-500 py-10">
              Failed to load doctor details. Please try again later.
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="pt-20">
        <div className="container py-12">
          <Button
            variant="ghost"
            className="mb-6 flex items-center"
            onClick={() => navigate(-1)}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Doctors
          </Button>

          <div className="max-w-4xl mx-auto">
            <motion.div
              className="grid md:grid-cols-3 gap-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="md:col-span-1">
                <img
                  src={doctor.avatarUrl || "/placeholder.svg"}
                  alt={`${doctor.firstName || ""} ${doctor.lastName || ""}`}
                  className="w-full rounded-lg shadow-lg mb-4"
                />
              </div>

              <div className="md:col-span-2">
                <h1 className="text-3xl font-bold mb-2">
                  {doctor.firstName && doctor.lastName
                    ? `Dr. ${doctor.firstName} ${doctor.lastName}`
                    : doctor.userName}
                </h1>
                <p className="text-primary text-xl mb-4">Dermatologist</p>

                <div className="grid gap-4 mb-8">
                  <div>
                    <h3 className="font-semibold">Experience</h3>
                    <p className="text-muted-foreground">
                      {doctor.experience
                        ? `${doctor.experience} years`
                        : "Not specified"}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-semibold">Education</h3>
                    <p className="text-muted-foreground">
                      {doctor.education || "Not specified"}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-semibold">Languages</h3>
                    <p className="text-muted-foreground">
                      {doctor.languages || "English"}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-semibold">Certifications</h3>
                    <p className="text-muted-foreground">
                      {doctor.certifications || "Not specified"}
                    </p>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-4 mt-6">
                  <Button
                    className="flex-1"
                    onClick={() => {
                      // Use the first available schedule if any exist
                      const firstSchedule =
                        schedules && schedules.length > 0 ? schedules[0] : null;
                      if (firstSchedule) {
                        handleScheduleSelect(firstSchedule);
                      }
                    }}
                  >
                    Book Appointment
                  </Button>
                  <Button variant="outline" className="flex-1">
                    Send Message
                  </Button>
                </div>
              </div>
            </motion.div>

            <motion.div
              className="mt-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold flex items-center">
                  <Calendar className="mr-2 h-6 w-6" />
                  Available Schedule
                </h2>
                <div className="flex gap-2">
                  {/* Filter dropdown */}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline">{getFilterLabel()}</Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => handleFilterChange("day")}
                      >
                        Day View
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleFilterChange("week")}
                      >
                        Week View
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleFilterChange("month")}
                      >
                        Month View
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => handleFilterChange(undefined)}
                      >
                        All Schedules
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>

                  {/* Navigation buttons */}
                  <Button
                    variant="outline"
                    onClick={() => setCurrentWeek(currentWeek - 1)}
                    disabled={currentWeek <= 0}
                  >
                    Previous {filter || "Period"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setCurrentWeek(currentWeek + 1)}
                  >
                    Next {filter || "Period"}
                  </Button>
                </div>
              </div>

              <div className="text-center mb-4">
                <p className="text-muted-foreground">
                  {format(startDate, "MMM d, yyyy")} -{" "}
                  {format(endDate, "MMM d, yyyy")}
                </p>
              </div>

              {schedulesError ? (
                <div className="text-center text-red-500 py-4">
                  Failed to load schedules. Please try again later.
                </div>
              ) : schedules?.length === 0 ? (
                <div className="text-center py-8 border rounded-lg">
                  <p className="text-muted-foreground">
                    No available schedules for this {filter || "period"}.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {schedulesByDay &&
                    Object.entries(schedulesByDay).map(
                      ([day, daySchedules]) => {
                        // Find earliest and latest times for this day
                        if (
                          !Array.isArray(daySchedules) ||
                          daySchedules.length === 0
                        )
                          return null;

                        const sortedSchedules = [...daySchedules].sort(
                          (a, b) =>
                            new Date(a.startTime).getTime() -
                            new Date(b.startTime).getTime()
                        );

                        return (
                          <div key={day} className="mb-6">
                            <h3 className="font-semibold text-lg mb-3">
                              {day}{" "}
                              {sortedSchedules.length > 0 && (
                                <span className="font-normal text-muted-foreground ml-2">
                                  {format(
                                    parseISO(sortedSchedules[0].startTime),
                                    "MMM d, yyyy"
                                  )}
                                </span>
                              )}
                            </h3>
                            <div className="grid grid-cols-2 gap-2">
                              {sortedSchedules.map((schedule) => (
                                <div
                                  key={schedule.id}
                                  className={`p-3 rounded-lg border bg-card w-full min-w-[150px] ${
                                    schedule.status === "BOOKED"
                                      ? "opacity-70 cursor-not-allowed"
                                      : "hover:shadow-md cursor-pointer"
                                  } transition-shadow`}
                                  onClick={() => {
                                    // Only allow selection if not booked
                                    if (schedule.status !== "BOOKED") {
                                      handleScheduleSelect(schedule);
                                    } else {
                                      toast({
                                        title: "Schedule unavailable",
                                        description:
                                          "This time slot is already booked.",
                                        variant: "destructive",
                                      });
                                    }
                                  }}
                                >
                                  <p className="text-muted-foreground whitespace-nowrap">
                                    {format(
                                      parseISO(schedule.startTime),
                                      "h:mm a"
                                    )}{" "}
                                    -{" "}
                                    {format(
                                      parseISO(schedule.endTime),
                                      "h:mm a"
                                    )}
                                  </p>
                                  {schedule.status === "BOOKED" && (
                                    <p className="text-red-500 mt-1 text-sm font-medium">
                                      Booked
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      }
                    )}
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </main>

      {/* Appointment Notes Modal */}
      <Dialog open={isModalOpen} onOpenChange={handleModalClose}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Book Appointment</DialogTitle>
            <DialogDescription>
              {selectedSchedule && (
                <>
                  Appointment with Dr. {doctor.firstName} {doctor.lastName} on{" "}
                  {format(parseISO(selectedSchedule.startTime), "EEEE, MMMM d")}{" "}
                  at {format(parseISO(selectedSchedule.startTime), "h:mm a")}
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="notes">Additional Notes</Label>
              <Textarea
                id="notes"
                placeholder="Please describe your symptoms or any specific concerns..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={5}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                // Clear schedule ID from URL
                window.history.pushState({}, "", window.location.pathname);
                setIsModalOpen(false);
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleBookAppointment}>Continue</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default DoctorProfile;
