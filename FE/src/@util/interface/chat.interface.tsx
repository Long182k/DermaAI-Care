import { Socket } from "socket.io-client";
import { User } from "../types/auth.type";

export interface Participant {
  id: string;
  userId: string;
  chatRoomId: string;
  joinedAt: string;
}

export interface CreateDirectChatResponse {
  id: string;
  name: string;
  type: "DIRECT" | "GROUP";
  creatorId: string;
  createdAt: string;
  updatedAt: string;
  userId: string | null;
  participants: Participant[];
}

export interface SendMessageResponse {
  id: string;
  content: string;
  type: "MESSAGE" | "image" | "video";
  senderId: string;
  receiverId: string;
  chatRoomId: string;
  createdAt: string;
  updatedAt: string;
}

export interface ChatParticipant {
  id: string;
  userId: string;
  chatRoomId: string;
  joinedAt: string;
  user?: {
    id: string;
    userName: string;
  };
}

export interface ChatMessageResponse {
  id: string;
  content: string;
  type: "MESSAGE" | "image" | "video";
  senderId: string;
  receiverId: string;
  chatRoomId: string;
  createdAt: string;
  updatedAt: string;
}

export interface ChatRoom {
  id: string;
  name: string;
  type: "DIRECT" | "GROUP";
  creatorId: string;
  createdAt: string; 
  updatedAt: string;
  userId: string | null;
  participants: ChatParticipant[];
  messages?: ChatMessageResponse[];
  user?: User | null; 
}

export interface ChatRoomResponse {
  data: ChatRoom[];
}

export interface GetMessageResponse {
  id?: string;
  content: string;
  type?: "MESSAGE" | "image" | "video";
  senderId: string;
  receiverId: string;
  chatRoomId?: string;
  createdAt?: string;
  updatedAt?: string;
  user?: {
    id: string;
    userName: string;
    email: string;
    role: "USER" | "ADMIN";
    displayName: string | null;
    hashedPassword: string;
    hashedRefreshToken: string;
    avatarUrl: string | null;
    bio: string | null;
    isActive: boolean;
    createdAt: string;
  };
}

export interface ChatStore {
  messages: GetMessageResponse[];
  selectedChatRoom: null | ChatRoom;
  isUsersLoading: boolean;
  isMessagesLoading: boolean;

  fetchContacts: () => Promise<User[]>;
  getMessages: (chatRoomId: string) => Promise<GetMessageResponse[]>;
  getChatRoom: (currentUserId: string) => Promise<ChatRoom[]>;

  sendMessage: (messageData: {
    chatRoomId: string;
    senderId: string;
    receiverId: string;
    content: string;
  }) => Promise<void>;

  createDirectChat: (directChatData: {
    senderId: string;
    receiverId: string;
    type: "DIRECT" | "GROUP";
    name: string;
  }) => Promise<CreateDirectChatResponse>;

  subscribeToMessages: () => void;
  unsubscribeFromMessages: () => void;
  setSelectedChatRoom: (selectedChatRoom: ChatRoom | null) => void;
}

export interface ClientToServerEvents {
  sendMessage: (message: string) => void;
}

export interface ServerToClientEvents {
  getOnlineUsers: (userIds: string[]) => void;
  newMessage: (message: GetMessageResponse) => void;
}

export type SocketInstance = Socket<ServerToClientEvents, ClientToServerEvents>;
