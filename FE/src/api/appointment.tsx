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
  languages?: string;
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

export type ScheduleFilter = "day" | "week" | "month" | undefined;

export interface CreateAppointmentDto {
  patientId: string;
  scheduleId: string;
  notes?: string;
}

export const doctorApi = {
  getAllDoctors: async (page: number = 1, limit: number = 100) => {
    const response = await axiosClient.get(
      `/users/doctors?page=${page}&limit=${limit}`
    );
    return response.data;
  },

  getDoctorById: async (doctorId: string) => {
    const response = await axiosClient.get(`/users/doctor/${doctorId}`);
    return response.data;
  },

  getDoctorSchedules: async (
    doctorId: string,
    startDate: Date,
    endDate: Date,
    filter?: ScheduleFilter
  ) => {
    const url = `/schedules/doctor/${doctorId}?startDate=${startDate.toISOString()}&endDate=${endDate.toISOString()}${
      filter ? `&filter=${filter}` : ""
    }`;
    const response = await axiosClient.get(url);
    return response.data;
  },
};

export interface Appointment {
  id: string;
  patientId: string;
  doctorId: string;
  status: string;
  notes?: string;
  createdAt: string;
  updatedAt: string;
  scheduleId: string;
  Patient: {
    id: string;
    userName: string;
    firstName: string | null;
    lastName: string | null;
    email: string;
    avatarUrl?: string;
  };
  Doctor: {
    id: string;
    userName: string;
    firstName: string | null;
    lastName: string | null;
    email: string;
    education?: string;
    avatarUrl?: string;
  };
  Schedule: {
    id: string;
    doctorId: string;
    startTime: string;
    endTime: string;
    status: string;
    createdAt: string;
    updatedAt: string;
  };
}

// Add this method to the appointmentApi object
export const appointmentApi = {
  createAppointment: async (data: CreateAppointmentDto) => {
    const response = await axiosClient.post("/appointments", data);
    return response.data;
  },

  getUserAppointments: async () => {
    const response = await axiosClient.get("/appointments/user");
    return response.data as Appointment[];
  },

  getAppointmentById: async (id: string) => {
    const response = await axiosClient.get(`/appointments/${id}`);
    return response.data as Appointment;
  },

  checkoutSession: async (id: string) => {
    const response = await axiosClient.post(
      `/payment/create-checkout-session/${id}`
    );
    return response.data;
  },

  cancelAppointment: async (id: string) => {
    const response = await axiosClient.patch(`/appointments/${id}/cancel`);
    return response.data;
  },
};
