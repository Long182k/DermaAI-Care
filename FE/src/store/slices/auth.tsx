import { toast } from "react-toastify";
import { StateCreator } from "zustand";
import { persist, PersistOptions } from "zustand/middleware";
import {
  AuthStore,
  LoginResponse,
  RegisterResponse,
} from "../../@util/interface/auth.interface";
import {
  LoginParams,
  RegisterNewUserParams,
  User,
} from "../../@util/types/auth.type";
import { axiosInitialClient } from "../../api/axiosConfig";

// Configure persist options for AuthStore
const authPersistOptions: PersistOptions<
  AuthStore,
  Pick<AuthStore, "userInfo" | "onlineUsers">
> = {
  name: "auth_storage",
  partialize: (state) => ({
    userInfo: state.userInfo,
    onlineUsers: state.onlineUsers,
  }),
};

// Create AuthStore logic
const createAuthState: StateCreator<AuthStore> = (set, get) => ({
  accessToken: undefined,
  userInfo: {} as User,
  socket: null,
  onlineUsers: [],
  addAccessToken: (accessToken: string) => set({ accessToken }),
  getAccessToken: () => get().accessToken,
  removeAccessToken: () => set({ accessToken: undefined }),
  addUserInfo: (userInfo: User) => set({ userInfo }),
  getUserInfo: () => get().userInfo,
  removeUserInfo: () =>
    set({
      userInfo: undefined,
    }),
  signup: async (data: RegisterNewUserParams): Promise<RegisterResponse> => {
    try {
      const { data: response } = await axiosInitialClient.post(
        "/auth/register",
        data
      );
      console.log("🚀 response:", response);
      set({ userInfo: response });
      localStorage.setItem("access_token", response.accessToken);
      return response;
    } catch (error) {
      console.log("error", error);
      throw error;
    }
  },
  login: async (data: LoginParams): Promise<LoginResponse> => {
    console.log("data login", data);
    try {
      const { data: dataResponse } = await axiosInitialClient.post(
        "/auth/login",
        data
      );
      localStorage.setItem("access_token", dataResponse.accessToken);

      set({ userInfo: dataResponse });

      return dataResponse;
    } catch (error) {
      console.error("Login error", error);
      throw error; // Rejects the Promise with the error
    }
  },

  logout: async () => {
    const { userInfo } = get();

    try {
      await axiosInitialClient.post(`/auth/logout/${userInfo.userId}`);

      set({
        userInfo: undefined,
      });

      localStorage.removeItem("access_token");

      toast.success("Logged out successfully");
      get().removeUserInfo();
    } catch (error) {
      console.log("error", error);
    }
  },
});

export const createAuthStore = persist(createAuthState, authPersistOptions);
