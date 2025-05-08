
import { Navbar } from "@/components/Navbar";
import { Phone, Mail, MapPin } from "lucide-react";
import { motion } from "framer-motion";

const Contact = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="pt-20">
        <div className="container py-12">
          <motion.div 
            className="max-w-4xl mx-auto text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-bold mb-4">Contact Us</h1>
            <p className="text-lg text-muted-foreground">
              We're here to help! Reach out to us through any of the following channels.
            </p>
          </motion.div>

          <motion.div 
            className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div className="text-center p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
              <Phone className="h-12 w-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Phone</h3>
              <p className="text-muted-foreground">+1 (555) 123-4567</p>
              <p className="text-muted-foreground">Monday - Friday: 9 AM - 6 PM</p>
            </div>

            <div className="text-center p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
              <Mail className="h-12 w-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Email</h3>
              <p className="text-muted-foreground">support@dermaicare.com</p>
              <p className="text-muted-foreground">Response within 24 hours</p>
            </div>

            <div className="text-center p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
              <MapPin className="h-12 w-12 text-primary mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Office</h3>
              <p className="text-muted-foreground">International University</p>
              <p className="text-muted-foreground">Quarter 6, Linh Trung Ward, Thu Duc District, HCMC</p>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default Contact;
