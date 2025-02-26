import { useState } from "react";
import { Calendar } from "@/components/ui/calendar";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";

// Mock data for doctors (replace with actual API call later)
const doctors = [
  { id: 1, name: "Dr. Sarah Johnson", specialty: "Dermatologist" },
  { id: 2, name: "Dr. Michael Chen", specialty: "Dermatologist" },
  { id: 3, name: "Dr. Emily Brown", specialty: "Dermatologist" },
];

// Mock data for time slots (replace with actual API call later)
const timeSlots = [
  "09:00 AM",
  "10:00 AM",
  "11:00 AM",
  "02:00 PM",
  "03:00 PM",
  "04:00 PM",
];

export const BookingForm = () => {
  const [date, setDate] = useState<Date>();
  const [selectedDoctor, setSelectedDoctor] = useState("");
  const [selectedTime, setSelectedTime] = useState("");

  const handleBooking = () => {
    if (!date || !selectedDoctor || !selectedTime) {
      toast.error("Please fill in all fields");
      return;
    }

    // Here you would typically make an API call to save the booking
    toast.success("Appointment booked successfully!");
    console.log({
      date,
      doctorId: selectedDoctor,
      time: selectedTime,
    });
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div className="space-y-4">
        <label className="block text-lg font-medium">Select a Doctor</label>
        <Select value={selectedDoctor} onValueChange={setSelectedDoctor}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Choose your doctor" />
          </SelectTrigger>
          <SelectContent>
            {doctors.map((doctor) => (
              <SelectItem key={doctor.id} value={doctor.id.toString()}>
                {doctor.name} - {doctor.specialty}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-4">
        <label className="block text-lg font-medium">Select a Date</label>
        <Calendar
          mode="single"
          selected={date}
          onSelect={setDate}
          className="rounded-md border mx-auto"
          disabled={(date) => date < new Date()}
        />
      </div>

      <div className="space-y-4">
        <label className="block text-lg font-medium">Select a Time</label>
        <Select value={selectedTime} onValueChange={setSelectedTime}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Choose appointment time" />
          </SelectTrigger>
          <SelectContent>
            {timeSlots.map((time) => (
              <SelectItem key={time} value={time}>
                {time}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Button
        onClick={handleBooking}
        className="w-full"
        size="lg"
      >
        Book Appointment
      </Button>
    </div>
  );
};