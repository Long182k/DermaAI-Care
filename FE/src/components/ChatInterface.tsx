import {
  useChatContext,
  Channel,
  ChannelList,
  Window,
  ChannelHeader,
  MessageList,
  MessageInput,
  Thread,
} from "stream-chat-react";
import "stream-chat-react/dist/css/v2/index.css";
import { ChannelListCustom } from "./chat/ChannelListCustom";
import SharedMedia from "./chat/SharedMedia";

export const ChatInterface = () => {
  const { client } = useChatContext();

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
              <ChannelHeader />
              <MessageList />
              <MessageInput />
            </Window>
          </div>
          <Thread />
        </Channel>
      </div>
      <SharedMedia />
    </div>
  );
};
