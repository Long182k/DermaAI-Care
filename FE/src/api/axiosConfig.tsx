import axios from "axios";

const API_URL = import.meta.env.VITE_SERVER_URL;

export const axiosInitialClient = axios.create({
  baseURL: API_URL,
});

export const axiosClient = axios.create({
  baseURL: API_URL,
});

axiosClient.interceptors.request.use((config) => {
  const accessToken = localStorage.getItem("access_token");

  if (accessToken) {
    config.headers["Authorization"] = `Bearer ${accessToken}`;
  }
  return config;
});
