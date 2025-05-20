import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StreamVideo, StreamVideoClient } from "@stream-io/video-react-sdk";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import { StreamChat } from "stream-chat";
import { Chat as StreamChatComponent } from "stream-chat-react";
import About from "./pages/About";
import AdminDashboard from "./pages/AdminDashboard";
import Analysis from "./pages/Analysis";
import AppointmentsPage from "./pages/AppointmentsPage";
import Auth from "./pages/Auth";
import Booking from "./pages/Booking";
import Chat from "./pages/Chat";
import Contact from "./pages/Contact";
import DoctorProfile from "./pages/DoctorProfile";
import Doctors from "./pages/Doctors";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import PaymentPage from "./pages/PaymentPage";
import PredictionHistoryPage from "./pages/PredictionHistoryPage";
import Privacy from "./pages/Privacy";
import Services from "./pages/Services";
import { useAppStore } from "./store";
import UserProfile from "./pages/UserProfile";

// Initialize Stream Chat client
const chatClient = StreamChat.getInstance(import.meta.env.VITE_STREAM_KEY!);
const videoClient = new StreamVideoClient({
  apiKey: import.meta.env.VITE_STREAM_KEY!,
});

const queryClient = new QueryClient();
const App = () => {
  const { userInfo } = useAppStore();
  const [clientReady, setClientReady] = useState(false);
  const userId = userInfo?.userId ?? userInfo?.id;

  useEffect(() => {
    if (!userInfo) return;

    if (
      chatClient.tokenManager.token === userInfo.streamToken &&
      chatClient.userID === userId
    )
      return;

    // Connect user to Stream Chat
    const connectUser = async () => {
      try {
        const token = userInfo.streamToken;

        await chatClient.connectUser(
          {
            id: userId,
            name: userInfo.userName,
            image: userInfo.avatarUrl,
          },
          token
        );

        await videoClient.connectUser(
          {
            id: userId,
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
      videoClient.disconnectUser();
      setClientReady(false);
    };

    // Add userInfo.avatarUrl
  }, [userInfo, userId, userInfo?.streamToken, userInfo?.userName]);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <StreamVideo client={videoClient}>
            <StreamChatComponent client={chatClient}>
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/about" element={<About />} />
                <Route path="/services" element={<Services />} />
                <Route path="/doctors" element={<Doctors />} />
                <Route path="/profile" element={<UserProfile />} />
                <Route path="/admin-dashboard" element={<AdminDashboard />} />
                <Route path="/doctor/:id" element={<DoctorProfile />} />
                <Route
                  path="/book-appointment/:id"
                  element={<DoctorProfile />}
                />
                <Route path="/contact" element={<Contact />} />
                <Route path="/analysis" element={<Analysis />} />
                <Route path="/booking" element={<Booking />} />
                <Route path="/chat" element={<Chat />} />
                <Route path="/auth" element={<Auth />} />
                <Route path="/privacy" element={<Privacy />} />
                <Route path="/appointments" element={<AppointmentsPage />} />
                <Route path="/payment/:id" element={<PaymentPage />} />
                <Route
                  path="/prediction-history"
                  element={<PredictionHistoryPage />}
                />
                <Route
                  path="/prediction-history/:patientId"
                  element={<PredictionHistoryPage />}
                />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </StreamChatComponent>
          </StreamVideo>
        </BrowserRouter>
        <ToastContainer />
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
