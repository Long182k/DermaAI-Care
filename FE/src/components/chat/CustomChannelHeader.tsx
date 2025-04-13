import { User } from '@/@util/types/auth.type'
import { Channel } from 'stream-chat'
import VideoCallButton from './VideoCallButton'
type TProps = {
    userInfo: User,
    channel: Channel,
    handleStartCall: () => void
}
const CustomChannelHeader = ({userInfo, channel, handleStartCall}: TProps) => {
  return (
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
  )
}

export default CustomChannelHeader