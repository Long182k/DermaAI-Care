export interface ChatMessage {
  id: string;
  content: string;
  type: string;
  senderId: string;
  receiverId: string;
  chatRoomId: string;
  createdAt: string;
  updatedAt: string;
  user: {
    id: string;
    userName: string;
    avatarUrl?: string;
  };
  attachments: {
    id: string;
    type: "image" | "video";
    url: string;
    chatMessageId: string;
    createdAt: string;
  }[];
} 