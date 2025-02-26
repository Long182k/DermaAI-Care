import { Navbar } from "@/components/Navbar";
import { BookingForm } from "@/components/BookingForm";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { BackButton } from "@/components/common/BackButton";

const Booking = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-background">
      <BackButton route="services" />
      <main className="container mx-auto py-10">
        <h1 className="text-4xl font-bold mb-8 text-center">
          Book an Appointment
        </h1>
        <BookingForm />
      </main>
    </div>
  );
};

export default Booking;
