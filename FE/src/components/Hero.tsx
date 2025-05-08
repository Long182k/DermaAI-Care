import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import homepageSkin from "@/assets/images/homepage.jpg";

export const Hero = () => {
  const navigate = useNavigate();

  return (
    <div className="relative overflow-hidden bg-gradient-to-b from-primary/10 to-background pt-20 pb-32">
      <div className="container relative">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-6 text-center lg:text-left">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tighter">
              AI-Powered Skin Disease Detection & Expert Care
            </h1>
            <p className="text-lg text-muted-foreground md:text-xl">
              Get instant AI analysis of skin conditions and connect with
              certified dermatologists for professional care.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Button
                size="lg"
                className="bg-primary hover:bg-primary/90"
                onClick={() => navigate("/services")}
              >
                Our Services <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </div>
          <div className="relative animate-float">
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/30 to-secondary/30 rounded-full blur-3xl" />
            <img
              src={homepageSkin}
              alt="Medical Illustration"
              className="relative w-full max-w-[600px] mx-auto"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
