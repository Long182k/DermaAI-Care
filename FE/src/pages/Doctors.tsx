import { Doctor, doctorApi } from "@/api/appointment";
import { Navbar } from "@/components/Navbar";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Doctors = () => {
  const navigate = useNavigate();

  const {
    data: doctors,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["doctors"],
    queryFn: () => doctorApi.getAllDoctors(),
    select: (res) => res.doctors,
  });

  // Function to handle doctor card click
  const handleDoctorClick = (doctorId: string) => {
    navigate(`/doctor/${doctorId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="pt-20">
        <div className="container py-12">
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold mb-4">Our Expert Doctors</h1>
            <p className="text-lg text-muted-foreground">
              Meet our team of certified dermatologists ready to help you
            </p>
          </motion.div>

          {isLoading ? (
            <div className="flex justify-center items-center py-20">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : error ? (
            <div className="text-center text-red-500 py-10">
              Failed to load doctors. Please try again later.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {doctors.map((doctor: Doctor, index: number) => (
                <motion.div
                  key={doctor.id}
                  className="rounded-lg border bg-card overflow-hidden hover:shadow-lg transition-all cursor-pointer"
                  onClick={() => handleDoctorClick(doctor.id)}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.2 }}
                >
                  <img
                    src={doctor.avatarUrl || "/placeholder.svg"}
                    alt={`${doctor.firstName || ""} ${doctor.lastName || ""}`}
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-6">
                    <h3 className="text-xl font-semibold mb-2">
                      {doctor.firstName && doctor.lastName
                        ? `Dr. ${doctor.firstName} ${doctor.lastName}`
                        : doctor.userName}
                    </h3>
                    <p className="text-primary mb-2">Dermatologist</p>
                    <p className="text-muted-foreground mb-4">
                      {doctor.experience
                        ? `${doctor.experience} years experience`
                        : "Experience not specified"}
                    </p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <span className="text-yellow-400">★</span>
                        <span className="ml-1 font-semibold">
                          {/* Placeholder for rating */}
                          4.8
                        </span>
                        <span className="text-muted-foreground ml-1">
                          {/* Placeholder for reviews */}
                          (96 reviews)
                        </span>
                      </div>
                      <button
                        className="text-primary hover:underline"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDoctorClick(doctor.id);
                        }}
                      >
                        View Profile
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Doctors;
