import { User } from "@/@util/types/auth.type";
import { Channel } from "stream-chat";
import VideoCallButton from "./VideoCallButton";
type TProps = {
  userInfo: User;
  channel: Channel;
  handleStartCall: () => void;
};
const CustomChannelHeader = ({
  userInfo,
  channel,
  handleStartCall,
}: TProps) => {
  // Find the opposite user in the channel members
  const getOppositeUser = () => {
    if (!channel?.state?.members) return null;

    // Find the member that is not the current user
    const oppositeUserEntry = Object.entries(channel.state.members).find(
      ([_, memberData]) =>
        memberData.user?.id !== userInfo.userId &&
        memberData.user?.id !== userInfo.id
    );

    return oppositeUserEntry ? oppositeUserEntry[1].user : null;
  };

  const oppositeUser = getOppositeUser();

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200">
      <div className="flex items-center">
        <img
          src={oppositeUser?.image || "/placeholder.svg"}
          alt="User Avatar"
          className="w-10 h-10 rounded-full mr-3"
        />
        <div>
          <div className="font-semibold">{oppositeUser?.name || "User"}</div>
          <div className="text-sm text-green-500">Online</div>
        </div>
      </div>
      <div className="flex">
        <VideoCallButton onStartCall={handleStartCall} />
      </div>
    </div>
  );
};

export default CustomChannelHeader;
