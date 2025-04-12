export type RegisterNewUserParams = {
  userName: string;
  firstName?: string;
  lastName?: string;
  email: string;
  phoneNumber?: string;
  password: string;
  gender: string;
  dateOfBirth: Date;
  role: string;
};

export type LoginParams = {
  username?: string;
  email?: string;
  password: string;
};

type ROLE = "USER" | "ADMIN";

export type User = {
  id: string;
  userId: string;
  userName: string;
  email: string;
  role: ROLE;
  displayName: string | null;
  hashedPassword: string;
  hashedRefreshToken: string;
  avatarUrl: string | null;
  coverPageUrl: string | null;
  bio: string | null;
  isActive: boolean;
  createdAt: string;
  dateOfBirth: string | null;
  accessToken?: string;
  refreshToken?: string;
  streamToken?: string | undefined | null;
};
