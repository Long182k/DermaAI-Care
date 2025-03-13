import { create } from "zustand";
import { AuthStore } from "../@util/interface/auth.interface";
import { createAuthStore } from "./slices/auth";
import { createPostStore, PostStore } from "./slices/post";

export const useAppStore = create<AuthStore & PostStore>((set, get, api) => ({
  ...createAuthStore(set, get, api),
  ...createPostStore(set, get, api),
}));
