export const getTextColor = (isDark: boolean) => ({
  color: isDark ? "#ffffff" : "#000000",
});

export const getBackgroundColor = (isDark: boolean) => ({
  background: isDark ? "#1f1f1f" : "#ffffff",
});

export const convertToHumanTime = (isoDate: string) => {
  const date = new Date(isoDate);

  const formattedDate = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);

  return formattedDate;
};

export const formatTimeAgo = (date: Date): string => {
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 5) {
    return "just now";
  }

  if (diffInSeconds < 60) {
    return `${diffInSeconds} seconds ago`;
  }

  if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes} ${minutes === 1 ? "minute" : "minutes"} ago`;
  }

  if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours} ${hours === 1 ? "hour" : "hours"} ago`;
  }

  if (diffInSeconds < 604800) {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days} ${days === 1 ? "day" : "days"} ago`;
  }

  if (diffInSeconds < 2592000) {
    const weeks = Math.floor(diffInSeconds / 604800);
    return `${weeks} ${weeks === 1 ? "week" : "weeks"} ago`;
  }

  if (diffInSeconds < 31536000) {
    const months = Math.floor(diffInSeconds / 2592000);
    return `${months} ${months === 1 ? "month" : "months"} ago`;
  }

  const years = Math.floor(diffInSeconds / 31536000);
  return `${years} ${years === 1 ? "year" : "years"} ago`;
};

export const isVideoUrl = (url: string): boolean => {
  return (
    url.match(/\.(mp4|webm|ogg)$/) !== null || url.includes("/video/upload/")
  );
};

export const formatDateTime = (dateTimeString: string): string => {
  const date = new Date(dateTimeString);

  // Format date: DD/MM/YYYY
  const formattedDate = date.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });

  // Format time: HH:mm
  const formattedTime = date.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  return `${formattedTime} ${formattedDate}`;
};

export function capitalizeFirstLetter(val: string) {
  return String(val).charAt(0).toUpperCase() + String(val).slice(1);
}

export function capitalizeWords(str: string) {
  return str
    .split(" ")
    .map((word) => capitalizeFirstLetter(word))
    .join(" ");
}