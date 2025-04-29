import { useState } from "react";
import { Call, useStreamVideoClient } from "@stream-io/video-react-sdk";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Phone, PhoneOff } from "lucide-react";

interface IncomingCallNotificationProps {
  channelId: string;
  callerName: string;
  onAccept: (call: Call) => void;
  onDecline: () => void;
}

export default function IncomingCallNotification({
  channelId,
  callerName,
  onAccept,
  onDecline,
}: IncomingCallNotificationProps) {
  const [isOpen, setIsOpen] = useState(true);
  const videoClient = useStreamVideoClient();

  const handleAccept = async () => {
    if (!videoClient) return;
    // default_5aae7163-5370-4fe8-9e0e-64e33e3e1ed1
    try {
      // Get the call
      const call = videoClient.call("default", channelId);
      await call.join({ create: true });

      // Close notification and pass call to parent without joining
      setIsOpen(false);
      onAccept(call);
    } catch (error) {
      console.error("Error accepting call:", error);
    }
  };

  const handleDecline = async () => {
    setIsOpen(false);
    onDecline();
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="sm:max-w-md">
        <div className="flex flex-col items-center space-y-4 p-6">
          <div className="text-xl font-semibold">Incoming Call</div>
          <div className="text-lg">{callerName} is calling you</div>
          <div className="flex space-x-4 mt-4">
            <Button
              onClick={handleAccept}
              className="bg-green-500 hover:bg-green-600"
            >
              <Phone className="mr-2 h-5 w-5" />
              Accept
            </Button>
            <Button onClick={handleDecline} variant="destructive">
              <PhoneOff className="mr-2 h-5 w-5" />
              Decline
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
