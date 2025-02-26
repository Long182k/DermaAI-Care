import { Navbar } from "@/components/Navbar";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { BackButton } from "@/components/common/BackButton";

const doctors = [
  {
    id: "1",
    name: "Dr. Sarah Johnson",
    specialty: "Dermatologist",
    experience: "15 years",
    image: "/placeholder.svg",
    rating: 4.9,
    reviews: 128,
  },
  {
    id: "2",
    name: "Dr. Michael Chen",
    specialty: "Dermatologist",
    experience: "12 years",
    image: "/placeholder.svg",
    rating: 4.8,
    reviews: 96,
  },
  {
    id: "3",
    name: "Dr. Emily Williams",
    specialty: "Dermatologist",
    experience: "10 years",
    image: "/placeholder.svg",
    rating: 4.7,
    reviews: 84,
  },
];

const Doctors = () => {
  const navigate = useNavigate();

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

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {doctors.map((doctor, index) => (
              <motion.div
                key={doctor.id}
                className="rounded-lg border bg-card overflow-hidden hover:shadow-lg transition-all cursor-pointer"
                onClick={() => navigate(`/doctors/${doctor.id}`)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <img
                  src={doctor.image}
                  alt={doctor.name}
                  className="w-full h-48 object-cover"
                />
                <div className="p-6">
                  <h3 className="text-xl font-semibold mb-2">{doctor.name}</h3>
                  <p className="text-primary mb-2">{doctor.specialty}</p>
                  <p className="text-muted-foreground mb-4">
                    {doctor.experience} experience
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-yellow-400">★</span>
                      <span className="ml-1 font-semibold">
                        {doctor.rating}
                      </span>
                      <span className="text-muted-foreground ml-1">
                        ({doctor.reviews} reviews)
                      </span>
                    </div>
                    <button
                      className="text-primary hover:underline"
                      onClick={() => navigate(`/doctors/${doctor.id}`)}
                    >
                      View Profile
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Doctors;
