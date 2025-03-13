import { Post } from "./post.type";

export type Bookmark = {
  createdAt: string;
  id: string;
  postId: string;
  userId: string;
  post: Post;
};

export type BookmarkResponse = {
  data: Bookmark[];
  meta: {
    total: number;
    page: number;
    limit: number;
    totalPages: number;
  };
};
