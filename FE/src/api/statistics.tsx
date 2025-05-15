import { axiosClient } from "./axiosConfig";
const API_URL = import.meta.env.VITE_SERVER_URL;

export interface OverviewStats {
  summary: {
    totalPatients: number;
    totalDoctors: number;
    totalPredictions: number;
    totalAppointments: number;
    totalPayments: number;
  };
  charts: {
    monthly: any;
  };
}

export interface PatientStats {
  summary: {
    totalPatients: number;
    genderDistribution: any;
  };
  charts: {
    monthlyRegistrations: {
      labels: string[];
      data: number[];
    };
    topPatientsByAppointments: Array<{
      name: string;
      count: number;
    }>;
    topPatientsByPredictions: Array<{
      name: string;
      count: number;
    }>;
  };
}

export interface DoctorStats {
  summary: {
    totalDoctors: number;
    genderDistribution: any;
  };
  charts: {
    topDoctorsByAppointments: Array<{
      name: string;
      count: number;
    }>;
    doctorsByExperience: any;
    doctorsByLanguages: any;
  };
}

export interface PredictionStats {
  summary: {
    totalPredictions: number;
    predictionsByStatus: any;
  };
  charts: {
    monthlyPredictions: {
      labels: string[];
      data: number[];
    };
    predictionsByType: any;
  };
}

export interface AppointmentStats {
  summary: {
    totalAppointments: number;
    appointmentsByStatus: any;
  };
  charts: {
    monthlyAppointments: {
      labels: string[];
      data: number[];
    };
    appointmentsByDay: any;
    appointmentsByTime: any;
  };
}

export interface PaymentStats {
  summary: {
    totalPayments: number;
    completedPayments: number;
    pendingPayments: number;
    totalAmountCompleted: number;
    paymentsByStatus: Array<{
      status: string;
      count: number;
      totalAmount: number;
    }>;
  };
  charts: {
    paymentsByDay: Array<{
      day: string;
      count: number;
      totalAmount: number;
    }>;
    monthlyPayments: {
      labels: string[];
      data: number[];
    };
  };
}

export interface EditUserNamesDto {
  firstName?: string;
  lastName?: string;
}

export interface ChangeUserActiveDto {
  isActive: boolean;
}

// API functions
export const statisticsApi = {
  getOverview: async (): Promise<OverviewStats> => {
    const response = await axiosClient.get(`${API_URL}/statistics/overview`);
    return response.data;
  },

  getPatients: async (): Promise<PatientStats> => {
    const response = await axiosClient.get(`${API_URL}/statistics/patients`);
    return response.data;
  },

  getDoctors: async (): Promise<DoctorStats> => {
    const response = await axiosClient.get(`${API_URL}/statistics/doctors`);
    return response.data;
  },

  getPredictions: async (): Promise<PredictionStats> => {
    const response = await axiosClient.get(`${API_URL}/statistics/predictions`);
    return response.data;
  },

  getAppointments: async (): Promise<AppointmentStats> => {
    const response = await axiosClient.get(
      `${API_URL}/statistics/appointments`
    );
    return response.data;
  },

  getPayments: async (): Promise<PaymentStats> => {
    const response = await axiosClient.get(`${API_URL}/statistics/payments`);
    return response.data;
  },

  editUserNames: async (
    userId: string,
    dto: EditUserNamesDto
  ): Promise<any> => {
    const response = await axiosClient.patch(
      `${API_URL}/users/admin/edit-names/${userId}`,
      dto
    );
    return response.data;
  },

  changeUserActive: async (userId: string, isActive: boolean): Promise<any> => {
    const response = await axiosClient.patch(
      `${API_URL}/users/admin/change-active/${userId}`,
      { isActive }
    );
    return response.data;
  },

  
};
