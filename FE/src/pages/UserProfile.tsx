import { useAppStore } from "@/store";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Navbar } from "@/components/Navbar";
import { format, parseISO } from "date-fns";
import { Calendar, FileImage } from "lucide-react";

const UserProfile = () => {
  const { userInfo } = useAppStore();
  const navigate = useNavigate();

  const formatDateOfBirth = (dob?: string) => {
    if (!dob) return "N/A";
    try {
      const date = typeof dob === "string" ? parseISO(dob) : dob;
      return format(date, "MMMM d, yyyy");
    } catch {
      return dob;
    }
  };

  if (!userInfo) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <p className="text-lg">No user information available.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar />
      <div className="flex flex-1 items-center justify-center">
        <div className="bg-white rounded-2xl shadow-xl p-12 w-full max-w-2xl flex flex-col items-center border border-gray-100">
          <img
            src={userInfo.avatarUrl}
            alt="User Avatar"
            className="w-36 h-36 rounded-full mb-6 object-cover border-4 border-primary/30 shadow"
          />
          <h2 className="text-3xl font-bold mb-2">{userInfo.userName}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-2 w-full max-w-lg mb-6 mt-4">
            <div className="text-gray-700">
              <span className="font-semibold">Date of Birth:</span>{" "}
              {formatDateOfBirth(userInfo.dateOfBirth)}
            </div>
            <div className="text-gray-700">
              <span className="font-semibold">Phone:</span>{" "}
              {userInfo.phoneNumber || "N/A"}
            </div>
            <div className="text-gray-700">
              <span className="font-semibold">Gender:</span>{" "}
              {userInfo.gender || "N/A"}
            </div>
            <div className="text-gray-700">
              <span className="font-semibold">Email:</span> {userInfo.email}
            </div>
          </div>
          <div className="flex gap-6 mt-6 w-full max-w-lg">
            <Button
              className="flex-1 py-4 text-lg"
              onClick={() => navigate("/appointments")}
            >
              <Calendar className="h-5 w-5 mr-2" />
              Appointments
            </Button>
            <Button
              className="flex-1 py-4 text-lg"
              variant="outline"
              onClick={() => navigate("/prediction-history")}
            >
              <FileImage className="h-5 w-5 mr-2" />
              Prediction History
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
