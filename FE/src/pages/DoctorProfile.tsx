
import { Navbar } from "@/components/Navbar";
import { useParams } from "react-router-dom";
import { Calendar } from "lucide-react";
import { motion } from "framer-motion";

// Mock data - in a real app, this would come from an API
const doctorDetails = {
  "1": {
    name: "Dr. Sarah Johnson",
    specialty: "Dermatologist",
    age: 42,
    experience: "15 years",
    education: "MD from Stanford University",
    certifications: ["American Board of Dermatology", "Fellowship in Cosmetic Dermatology"],
    languages: ["English", "Spanish"],
    image: "/placeholder.svg",
    rating: 4.9,
    reviews: 128,
    schedule: [
      { day: "Monday", hours: "9:00 AM - 5:00 PM" },
      { day: "Tuesday", hours: "9:00 AM - 5:00 PM" },
      { day: "Wednesday", hours: "9:00 AM - 5:00 PM" },
      { day: "Thursday", hours: "9:00 AM - 5:00 PM" },
      { day: "Friday", hours: "9:00 AM - 3:00 PM" }
    ]
  },
  // ... add other doctor details
};

const DoctorProfile = () => {
  const { id } = useParams();
  const doctor = doctorDetails[id as keyof typeof doctorDetails];

  if (!doctor) {
    return <div>Doctor not found</div>;
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="pt-20">
        <div className="container py-12">
          <div className="max-w-4xl mx-auto">
            <motion.div 
              className="grid md:grid-cols-3 gap-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="md:col-span-1">
                <img 
                  src={doctor.image} 
                  alt={doctor.name}
                  className="w-full rounded-lg shadow-lg mb-4"
                />
                <div className="flex items-center justify-center mb-4">
                  <span className="text-yellow-400 text-xl">★</span>
                  <span className="ml-1 font-semibold text-lg">{doctor.rating}</span>
                  <span className="text-muted-foreground ml-1">({doctor.reviews} reviews)</span>
                </div>
              </div>

              <div className="md:col-span-2">
                <h1 className="text-3xl font-bold mb-2">{doctor.name}</h1>
                <p className="text-primary text-xl mb-4">{doctor.specialty}</p>
                
                <div className="grid gap-4 mb-8">
                  <div>
                    <h3 className="font-semibold">Experience</h3>
                    <p className="text-muted-foreground">{doctor.experience}</p>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold">Education</h3>
                    <p className="text-muted-foreground">{doctor.education}</p>
                  </div>

                  <div>
                    <h3 className="font-semibold">Languages</h3>
                    <p className="text-muted-foreground">{doctor.languages.join(", ")}</p>
                  </div>

                  <div>
                    <h3 className="font-semibold">Certifications</h3>
                    <ul className="list-disc list-inside text-muted-foreground">
                      {doctor.certifications.map((cert, index) => (
                        <li key={index}>{cert}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </motion.div>

            <motion.div 
              className="mt-12"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                <Calendar className="mr-2 h-6 w-6" />
                Available Schedule
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {doctor.schedule.map((slot, index) => (
                  <div 
                    key={index}
                    className="p-4 rounded-lg border bg-card"
                  >
                    <h3 className="font-semibold mb-2">{slot.day}</h3>
                    <p className="text-muted-foreground">{slot.hours}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default DoctorProfile;
