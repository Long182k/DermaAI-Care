import { Navbar } from "@/components/Navbar";
import { ChatInterface } from "@/components/ChatInterface";
import { BackButton } from "@/components/common/BackButton";

const Chat = () => {
  return (
    <div className="min-h-screen bg-background">
      <BackButton route="services" />
      <main className="container mx-auto py-6">
        <ChatInterface />
      </main>
    </div>
  );
};

export default Chat;
