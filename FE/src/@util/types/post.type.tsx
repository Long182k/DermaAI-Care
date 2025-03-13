import { User } from "./auth.type";

export type MediaType = "image" | "video";

export interface Media {
  id: string;
  type: MediaType;
  url: string;
  postId?: string;
  createdAt: Date;
}

export interface Comment {
  id: string;
  content: string;
  createdAt: Date;
  userId: string;
  postId: string;
  sentiment: string;
  user: User;
}

export interface Like {
  id: string;
  userId: string;
  postId: string;
  createdAt: Date;
  user: User;
}

export interface Bookmark {
  id: string;
  userId: string;
  postId: string;
  createdAt: Date;
  user: User;
}

export interface Post {
  id: string;
  content: string;
  userId: string;
  sentiment: string;
  createdAt: Date;
  user: User;
  attachments: Media[];
  comments: Comment[];
  likes: Like[];
  bookmarks: Bookmark[];
  _count: {
    likes: number;
    comments: number;
    bookmarks: number;
  };
}

export interface CreatePostDto {
  content: string;
  attachments?: {
    type: MediaType;
    url: string;
  }[];
}

export interface UpdatePostDto {
  content?: string;
  attachments?: {
    type: MediaType;
    url: string;
  }[];
}

export interface PostResponse {
  data: Post[];
  meta: {
    total: number;
    page: number;
    limit: number;
    totalPages: number;
  };
}

export interface PostInteractionResponse {
  liked?: boolean;
  bookmarked?: boolean;
}

export interface ApiResponse<T> {
  data: T;
  meta?: {
    total: number;
    page: number;
    limit: number;
    totalPages: number;
  };
}

export type CommentMutationVariables = {
  id: string;
  content: string;
};

export type CreateCommentDto = {
  content: string;
  imageUrl?: string;
};
