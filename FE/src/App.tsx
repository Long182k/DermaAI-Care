import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import About from "./pages/About";
import Services from "./pages/Services";
import Doctors from "./pages/Doctors";
import DoctorProfile from "./pages/DoctorProfile";
import Contact from "./pages/Contact";
import Analysis from "./pages/Analysis";
import Booking from "./pages/Booking";
import Chat from "./pages/Chat";
import Auth from "./pages/Auth";
import Privacy from "./pages/Privacy";
import NotFound from "./pages/NotFound";
import AppointmentsPage from "./pages/AppointmentsPage";
import PaymentPage from "./pages/PaymentPage";
import AnalysisPage from "./pages/AnalysisPage";
import { useAppStore } from "./store";
import { StreamChat } from "stream-chat";
import { Chat as StreamChatComponent } from "stream-chat-react";
import { useEffect, useState } from "react";

// Initialize Stream Chat client
const chatClient = StreamChat.getInstance(import.meta.env.VITE_STREAM_KEY!);

const queryClient = new QueryClient();
const App = () => {
  const { userInfo } = useAppStore();
  const [clientReady, setClientReady] = useState(false);

  useEffect(() => {
    if (!userInfo) return;

    if (
      chatClient.tokenManager.token === userInfo.streamToken &&
      chatClient.userID === userInfo.userId
    )
      return;

    // Connect user to Stream Chat
    const connectUser = async () => {
      try {
        const token = userInfo.streamToken;

        await chatClient.connectUser(
          {
            id: userInfo.userId,
            name: userInfo.userName,
            image: userInfo.avatarUrl,
          },
          token
        );

        setClientReady(true);
      } catch (error) {
        console.error("Failed to connect user", error);
      }
    };

    if (!chatClient.userID) {
      connectUser();
    }

    return () => {
      // Disconnect when component unmounts
      chatClient.disconnectUser();
      setClientReady(false);
    };
  }, [
    userInfo,
    userInfo.avatarUrl,
    userInfo.id,
    userInfo.streamToken,
    userInfo.userName,
  ]);

  // if (!clientReady) return <div className="loading">Loading...</div>;

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <StreamChatComponent client={chatClient}>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/about" element={<About />} />
              <Route path="/services" element={<Services />} />
              <Route path="/doctors" element={<Doctors />} />
              <Route path="/doctor/:id" element={<DoctorProfile />} />
              <Route path="/book-appointment/:id" element={<DoctorProfile />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/booking" element={<Booking />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/auth" element={<Auth />} />
              <Route path="/privacy" element={<Privacy />} />
              <Route path="/appointments" element={<AppointmentsPage />} />
              <Route path="/payment/:id" element={<PaymentPage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </StreamChatComponent>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
