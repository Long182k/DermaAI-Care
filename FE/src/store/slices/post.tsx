import { StateCreator } from "zustand";
import { Post } from "../../@util/types/post.type";

export interface PostStore {
  selectedPost: Post | null;
  setSelectedPost: (post: Post | null) => void;
  isCreatePostModalOpen: boolean;
  setCreatePostModalOpen: (isOpen: boolean) => void;
}

export const createPostStore: StateCreator<PostStore> = (set) => ({
  selectedPost: null,
  setSelectedPost: (post) => set({ selectedPost: post }),
  isCreatePostModalOpen: false,
  setCreatePostModalOpen: (isOpen) => set({ isCreatePostModalOpen: isOpen }),
});
