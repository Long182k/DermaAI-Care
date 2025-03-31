import { Navbar } from "@/components/Navbar";
import { useParams, useNavigate } from "react-router-dom";
import { Calendar, ArrowLeft, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { doctorApi, Schedule, ScheduleFilter } from "@/api/appointment";
import { Button } from "@/components/ui/button";
import { format, parseISO, startOfWeek, addDays, addWeeks } from "date-fns";
import { useState } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const DoctorProfile = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [currentWeek, setCurrentWeek] = useState(0);
  const [filter, setFilter] = useState<ScheduleFilter>('week');

  // Get current date range for schedules
  const today = new Date();
  const startDate = addWeeks(startOfWeek(today), currentWeek);
  const endDate = addDays(startDate, 6);

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
    queryFn: () => doctorApi.getDoctorSchedules(id!, startDate, endDate, filter),
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
      case 'day':
        return 'Day View';
      case 'week':
        return 'Week View';
      case 'month':
        return 'Month View';
      default:
        return 'All Schedules';
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
                <div className="flex items-center justify-center mb-4">
                  <span className="text-yellow-400 text-xl">★</span>
                  <span className="ml-1 font-semibold text-lg">4.9</span>
                  <span className="text-muted-foreground ml-1">
                    (128 reviews)
                  </span>
                </div>
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
                    onClick={() => navigate(`/book-appointment/${doctor.id}`)}
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
                      <Button variant="outline">
                        {getFilterLabel()}
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => handleFilterChange('day')}>
                        Day View
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleFilterChange('week')}>
                        Week View
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleFilterChange('month')}>
                        Month View
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleFilterChange(undefined)}>
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
                    Previous {filter || 'Period'}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setCurrentWeek(currentWeek + 1)}
                  >
                    Next {filter || 'Period'}
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
                    No available schedules for this {filter || 'period'}.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {schedulesByDay && Object.entries(schedulesByDay).map(([day, daySchedules]) => {
                    // Find earliest and latest times for this day
                    if (!Array.isArray(daySchedules) || daySchedules.length === 0) return null;
                    
                    const sortedSchedules = [...daySchedules].sort((a, b) => 
                      new Date(a.startTime).getTime() - new Date(b.startTime).getTime()
                    );
                    
                    const earliestTime = sortedSchedules[0].startTime;
                    const latestTime = sortedSchedules[sortedSchedules.length - 1].endTime;
                    
                    return (
                      <div
                        key={day}
                        className="p-6 rounded-lg border bg-card hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => navigate(`/book-appointment/${doctor.id}?day=${day}`)}
                      >
                        <h3 className="font-semibold text-lg mb-2">{day}</h3>
                        <p className="text-muted-foreground">
                          {format(parseISO(earliestTime), "h:mm a")} - {format(parseISO(latestTime), "h:mm a")}
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default DoctorProfile;
