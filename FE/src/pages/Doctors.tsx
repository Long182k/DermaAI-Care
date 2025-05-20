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
                  <div className="w-full h-64 overflow-hidden relative bg-gradient-to-br from-blue-50 to-indigo-50">
                    <div className="absolute inset-0 opacity-20">
                      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                          <pattern id={`dots-${doctor.id}`} patternUnits="userSpaceOnUse" width="20" height="20">
                            <circle cx="10" cy="10" r="1.5" fill="#6366F1" />
                          </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill={`url(#dots-${doctor.id})`} />
                      </svg>
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-4/5 h-4/5 flex items-center justify-center">
                        <img
                          src={doctor.avatarUrl || "/placeholder.svg"}
                          alt={`${doctor.firstName || ""} ${doctor.lastName || ""}`}
                          className="max-w-full max-h-full object-cover object-center"
                        />
                      </div>
                    </div>
                  </div>
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
