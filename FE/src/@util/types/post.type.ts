export interface CreateCommentDto {
  content: string;
  files?: File[];
}

export interface Comment {
  id: string;
  content: string;
  createdAt: string;
  userId: string;
  sentiment: string;
  user: {
    id: string;
    userName: string;
    avatarUrl?: string;
  };
  attachments: {
    id: string;
    commentId: string | null;
    postId: string | null;
    type: "image" | "video";
    url: string;
    createdAt: string;
  }[];
}

export interface Post {
  id: string;
  content: string;
  createdAt: string;
  userId: string;
  sentiment: string;
  user: {
    id: string;
    userName: string;
    avatarUrl?: string;
  };
  _count: {
    likes: number;
    comments: number;
    bookmarks: number;
  };
  comments: {
    id: string;
    content: string;
    createdAt: string;
    userId: string;
    sentiment: string;
    user: {
      id: string;
      userName: string;
      avatarUrl?: string;
    };
    attachments: {
      id: string;
      url: string;
      type: "image" | "video";
    }[];
  }[];
  attachments?: {
    id: string;
    url: string;
    type: "image" | "video";
  }[];
}
