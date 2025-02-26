import { Navbar } from "@/components/Navbar";
import { motion } from "framer-motion";
import {
  BookOpen,
  Calendar,
  MessageSquare,
  Globe,
  Brain,
  Bell,
  Mail,
} from "lucide-react";

import bookingImg from "@/assets/images/booking.jpg";
import consultationImg from "@/assets/images/instant.jpg";
import educationImg from "@/assets/images/education.jpg";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 },
};

const About = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="pt-20 pb-16">
        <div className="container">
          <motion.div
            className="max-w-4xl mx-auto text-center mb-16"
            initial={fadeIn.initial}
            animate={fadeIn.animate}
            transition={fadeIn.transition}
          >
            <h1 className="text-4xl font-bold mb-6">
              Welcome to DermAI Care - Your Digital Healthcare Companion
            </h1>
            <p className="text-lg text-muted-foreground mb-8">
              Hello there! We're excited to introduce you to DermAI Care, your
              all-in-one digital health companion designed to make managing skin
              conditions easier and more convenient.
            </p>
          </motion.div>

          <div className="grid gap-12">
            <motion.section
              className="grid md:grid-cols-2 gap-8 items-center"
              initial={fadeIn.initial}
              animate={fadeIn.animate}
              transition={{ ...fadeIn.transition, delay: 0.2 }}
            >
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Calendar className="h-8 w-8 text-primary" />
                  <h2 className="text-2xl font-semibold">
                    Book Appointments in Advance
                  </h2>
                </div>
                <p className="text-muted-foreground">
                  Say goodbye to long wait times at the doctor's office. Simply
                  upload a photo of your skin condition, schedule an appointment
                  with an expert doctor ahead of time, and have peace of mind
                  knowing you're prepared for your visit.
                </p>
              </div>
              <div className="rounded-lg overflow-hidden shadow-lg">
                <img
                  src={bookingImg}
                  alt="Booking Interface"
                  className="w-full h-64 object-cover"
                />
              </div>
            </motion.section>

            <motion.section
              className="grid md:grid-cols-2 gap-8 items-center md:flex-row-reverse"
              initial={fadeIn.initial}
              animate={fadeIn.animate}
              transition={{ ...fadeIn.transition, delay: 0.3 }}
            >
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <MessageSquare className="h-8 w-8 text-primary" />
                  <h2 className="text-2xl font-semibold">
                    Instant Doctor Consultations
                  </h2>
                </div>
                <p className="text-muted-foreground">
                  If you need immediate assistance, just click "Message Now" or
                  start a video call – your doctor is ready to help you right
                  away.
                </p>
              </div>
              <div className="rounded-lg overflow-hidden shadow-lg">
                <img
                  src={consultationImg}
                  alt="Consultation Interface"
                  className="w-full h-64 object-cover"
                />
              </div>
            </motion.section>

            <motion.section
              className="grid md:grid-cols-2 gap-8 items-center"
              initial={fadeIn.initial}
              animate={fadeIn.animate}
              transition={{ ...fadeIn.transition, delay: 0.4 }}
            >
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <BookOpen className="h-8 w-8 text-primary" />
                  <h2 className="text-2xl font-semibold">
                    Patient Education at Your Fingertips
                  </h2>
                </div>
                <p className="text-muted-foreground">
                  Learn about common skin conditions, effective treatments, and
                  practical tips to keep you healthy. Our expertly crafted
                  guides are designed to be both informative and easy to
                  understand.
                </p>
              </div>
              <div className="rounded-lg overflow-hidden shadow-lg">
                <img
                  src={educationImg}
                  alt="Education Resources"
                  className="w-full h-64 object-cover"
                />
              </div>
            </motion.section>

            <motion.div
              className="grid md:grid-cols-2 lg:grid-cols-4 gap-8"
              initial={fadeIn.initial}
              animate={fadeIn.animate}
              transition={{ ...fadeIn.transition, delay: 0.5 }}
            >
              <div className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
                <Globe className="h-8 w-8 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">
                  Multi-Language Support
                </h3>
                <p className="text-muted-foreground">
                  Available in multiple languages to cater to a global audience.
                </p>
              </div>

              <div className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
                <Brain className="h-8 w-8 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">
                  AI-Powered Analysis
                </h3>
                <p className="text-muted-foreground">
                  Get instant AI analysis of skin conditions with
                  recommendations.
                </p>
              </div>

              <div className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
                <Bell className="h-8 w-8 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">
                  Smart Notifications
                </h3>
                <p className="text-muted-foreground">
                  Stay informed with timely alerts about upcoming appointments.
                </p>
              </div>

              <div className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all">
                <Mail className="h-8 w-8 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">Google Login</h3>
                <p className="text-muted-foreground">
                  Quick and secure access with your Google account.
                </p>
              </div>
            </motion.div>
          </div>

          <motion.div
            className="max-w-3xl mx-auto text-center mt-16"
            initial={fadeIn.initial}
            animate={fadeIn.animate}
            transition={{ ...fadeIn.transition, delay: 0.6 }}
          >
            <h2 className="text-3xl font-bold mb-4">
              Why Wait? Experience DermAI Care Today!
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Transform the way you approach managing skin conditions. With
              DermAI Care, healthcare is simple and accessible, ensuring you
              receive the care you need without unnecessary delays.
            </p>
            <p className="text-xl font-semibold text-primary">
              Start your journey with DermAI Care today – where every photo
              brings you closer to better health!
            </p>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default About;
