import { Bot, Calendar, MessageSquare, Shield } from "lucide-react";
import { useNavigate } from "react-router-dom";

const features = [
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
    title: "Secure Chat",
    description:
      "Communicate directly with healthcare providers through our secure platform",
    route: "/chat",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description:
      "Your data is protected with enterprise-grade security and encryption",
    route: "/privacy",
  },
];

export const Features = () => {
  const navigate = useNavigate();

  return (
    <div className="py-24 bg-background">
      <div className="container">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Why Choose DermAI Care?</h2>
          <p className="text-muted-foreground text-lg">
            Advanced technology meets professional healthcare
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="p-6 rounded-lg border bg-card hover:shadow-lg transition-all cursor-pointer"
            >
              <feature.icon className="h-12 w-12 text-primary mb-4" />
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
