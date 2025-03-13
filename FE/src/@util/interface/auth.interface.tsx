import { SCREEN_MODE } from "../constant/constant";
import { LoginParams, RegisterNewUserParams, User } from "../types/auth.type";

export interface LoginFormProp {
  onSwitchMode: (mode: SCREEN_MODE) => void;
}

export interface RegisterFormProp {
  onSwitchMode: (mode: SCREEN_MODE) => void;
}

export interface ErrorResponseData {
  message: string;
}

export type LoginResponse = {
  accessToken: string;
  refreshToken: string;
  userId: string;
  userName: string;
  role: string;
};

export type RegisterResponse = {
  id: string;
  userName: string;
  email: string;
  role: string;
  hashedRefreshToken: string | null;
  avatarUrl: string | null;
  coverPageUrl: string | null;
  bio: string | null;
  dateOfBirth: string | null;
  isActive: boolean;
  createdAt: string;
  accessToken: string;
  refreshToken: string;
};

export interface AuthStore {
  accessToken: string | undefined;
  userInfo: User;
  onlineUsers: User[];
  addAccessToken: (accessToken: string) => void;
  getAccessToken: () => string | undefined;
  removeAccessToken: () => void;
  addUserInfo: (userInfo: User) => void;
  getUserInfo: () => User;
  removeUserInfo: () => void;
  signup: (data: RegisterNewUserParams) => Promise<RegisterResponse>;
  login: (data: LoginParams) => Promise<LoginResponse>;
  logout: () => Promise<void>;
}
