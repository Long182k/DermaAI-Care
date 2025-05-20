import { Navbar } from "@/components/Navbar";
import { Bot, Calendar, MessageSquare } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const services = [
  {
    icon: Bot,
    title: "AI Analysis",
    description:
      "Get instant preliminary analysis of skin conditions using advanced AI technology",
    route: "/analysis",
  },
  {
    icon: Calendar,
    title: "Easy Booking",
    description:
      "Schedule appointments with certified dermatologists at your convenience",
    route: "/doctors",
  },
  {
    icon: MessageSquare,
    title: "Real-time Communication",
    description:
      "Communicate directly with healthcare providers through our secure platform",
    route: "/chat",
  },
];

const Services = () => {
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
            <h1 className="text-4xl font-bold mb-4">Our Services</h1>
            <p className="text-lg text-muted-foreground">
              Comprehensive healthcare solutions at your fingertips
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {services.map((service, index) => (
              <motion.div
                key={service.title}
                className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all cursor-pointer"
                onClick={() => navigate(service.route)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <service.icon className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">{service.title}</h3>
                <p className="text-muted-foreground">{service.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Services;
