import {
  useChatContext,
  Channel,
  ChannelList,
  Window,
  MessageList,
  MessageInput,
  Thread,
} from "stream-chat-react";
import "stream-chat-react/dist/css/v2/index.css";
import { ChannelListCustom } from "./chat/ChannelListCustom";
import SharedMedia from "./chat/SharedMedia";
import { useState, useEffect } from "react";
import { useStreamVideoClient, Call } from "@stream-io/video-react-sdk";
import VideoCallModal from "./chat/VideoCallModal";
import VideoCallButton from "./chat/VideoCallButton";
import IncomingCallNotification from "./chat/IncomingCallNotification";
import { useAppStore } from "@/store";

export const ChatInterface = () => {
  const { client, channel } = useChatContext();
  const [activeCall, setActiveCall] = useState<Call | null>(null);
  const [showCallModal, setShowCallModal] = useState(false);
  const videoClient = useStreamVideoClient();
  const [incomingCall, setIncomingCall] = useState<{
    callId: string;
    channelId: string;
    callerName: string;
  } | null>(null);
  const {userInfo} = useAppStore()

  // Listen for incoming call messages
  useEffect(() => {
    if (!channel) return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleNewMessage = (event: any) => {
      const message = event.message;
      
      // Check if this is a call notification
      if (message.call_notification) {
        if (message.call_notification.status === 'started') {
          // Don't show notification if we initiated the call
          if (message.user.id === client.userID) return;
          
          // Show incoming call notification
          setIncomingCall({
            callId: message.call_notification.call_id,
            channelId: channel.id,
            callerName: message.user.name || 'Someone',
          });
        } else if (message.call_notification.status === 'ended') {
          // If the call has ended and it matches our active call
          if (activeCall && message.call_notification.call_id === activeCall.id) {
            // Close the call modal and reset call state
            setShowCallModal(false);
            setActiveCall(null);
          }
        }
      }
    };

    // Subscribe to new messages
    channel.on('message.new', handleNewMessage);

    // Cleanup
    return () => {
      channel.off('message.new', handleNewMessage);
    };
  }, [channel, client.userID, activeCall]);

  const handleStartCall = async () => {
    if (!videoClient) return;

    try {
      // Create and join the call
      const call = videoClient.call("default", channel.id);

      // Set the call state
      setActiveCall(call);
      setShowCallModal(true);

      // Notify the other user about the call via chat
      if (channel) {
        await channel.sendMessage({
          text: "Started a video call",
          call_notification: {
            call_id: call.id,
            type: "video",
            status: "started",
          },
        });
      }
    } catch (error) {
      console.error("Error starting call:", error);
    }
  };

  const handleAcceptCall = (call: Call) => {
    setActiveCall(call);
    setShowCallModal(true);
    setIncomingCall(null);
  };

  const handleDeclineCall = async () => {
    setIncomingCall(null);
    
    // Optionally send a message that the call was declined
    if (channel && incomingCall) {
      await channel.sendMessage({
        text: "Declined the video call",
        call_notification: {
          call_id: incomingCall.callId,
          type: "video",
          status: "declined",
        },
      });
    }
  };

  return (
    <div className="flex h-screen w-full">
      <div className="flex w-full h-full grow">
        <ChannelList
          List={ChannelListCustom}
          sendChannelsToList
          filters={{ members: { $in: [client.userID] } }}
        />
        <Channel>
          <div className="min-w-[750px]">
            <Window>
              <div className="flex flex-col h-full">
                <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200">
                  <div className="flex items-center">
                    <img
                      src={userInfo.avatarUrl}
                      alt="User Avatar"
                      className="w-10 h-10 rounded-full mr-3"
                    />
                    <div>
                      <div className="font-semibold">{channel?.data?.name}</div>
                      <div className="text-sm text-green-500">Online</div>
                    </div>
                  </div>
                  <div className="flex">
                    <VideoCallButton onStartCall={handleStartCall} />
                  </div>
                </div>
                <MessageList />
                <MessageInput />
              </div>
            </Window>
          </div>
          <Thread />
        </Channel>
      </div>
      <SharedMedia />
      {showCallModal && activeCall && (
        <VideoCallModal call={activeCall} />
      )}
      {incomingCall && (
        <IncomingCallNotification
          channelId={incomingCall.channelId}
          callerName={incomingCall.callerName}
          onAccept={handleAcceptCall}
          onDecline={handleDeclineCall}
        />
      )}
    </div>
  );
};
