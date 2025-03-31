import { axiosClient } from "./axiosConfig";

export interface Doctor {
  id: string;
  userName: string;
  firstName?: string;
  lastName?: string;
  email: string;
  experience?: number;
  education?: string;
  certifications?: string;
  avatarUrl?: string;
  role: string;
}

export interface Schedule {
  id: string;
  doctorId: string;
  startTime: string;
  endTime: string;
  status: string;
  createdAt: string;
  updatedAt: string;
}

export type ScheduleFilter = 'day' | 'week' | 'month' | undefined;

export const doctorApi = {
  getAllDoctors: async (page: number = 1, limit: number = 10) => {
    const response = await axiosClient.get(
      `/users/doctors?page=${page}&limit=${limit}`
    );
    return response.data;
  },
  
  getDoctorById: async (doctorId: string) => {
    const response = await axiosClient.get(`/users/doctor/${doctorId}`);
    return response.data;
  },
  
  getDoctorSchedules: async (doctorId: string, startDate: Date, endDate: Date, filter?: ScheduleFilter) => {
    const url = `/schedules/doctor/${doctorId}?startDate=${startDate.toISOString()}&endDate=${endDate.toISOString()}${filter ? `&filter=${filter}` : ''}`;
    const response = await axiosClient.get(url);
    return response.data;
  },
};
