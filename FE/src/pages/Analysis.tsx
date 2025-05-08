import { AIAnalysis } from "@/components/AIAnalysis";
import { Navbar } from "@/components/Navbar";

const Analysis = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto pt-20">
        <AIAnalysis />
      </main>
    </div>
  );
};

export default Analysis;