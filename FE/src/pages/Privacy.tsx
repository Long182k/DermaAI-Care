
import { Shield, Lock, UserCheck, Database } from "lucide-react";
import { Navbar } from "@/components/Navbar";

const securityFeatures = [
  {
    icon: Lock,
    title: "End-to-End Encryption",
    description:
      "All communications and data transfers are protected with military-grade encryption, ensuring your information remains private and secure.",
  },
  {
    icon: Shield,
    title: "HIPAA Compliance",
    description:
      "Our platform adheres to strict healthcare privacy standards, maintaining full HIPAA compliance to protect your medical information.",
  },
  {
    icon: UserCheck,
    title: "Secure Authentication",
    description:
      "Multi-factor authentication and advanced session management keep your account protected from unauthorized access.",
  },
  {
    icon: Database,
    title: "Data Protection",
    description:
      "Your medical records and personal information are stored with enterprise-level security measures and regular security audits.",
  },
];

const Privacy = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="pt-20">
        <div className="container py-12">
          <div className="max-w-3xl mx-auto text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">Your Privacy & Security</h1>
            <p className="text-lg text-muted-foreground">
              At DermAI Care, we prioritize the security and confidentiality of your medical information
              with state-of-the-art protection measures.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            {securityFeatures.map((feature) => (
              <div
                key={feature.title}
                className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all"
              >
                <feature.icon className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>

          <div className="prose prose-lg max-w-3xl mx-auto">
            <h2>Our Commitment to Your Privacy</h2>
            <p>
              We understand the sensitive nature of medical information and take extensive measures to
              ensure your data remains confidential and secure. Our platform employs multiple layers of
              security:
            </p>
            <ul>
              <li>Regular security audits and penetration testing</li>
              <li>24/7 system monitoring for suspicious activities</li>
              <li>Strict access controls and authentication protocols</li>
              <li>Secure data centers with physical security measures</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Privacy;
