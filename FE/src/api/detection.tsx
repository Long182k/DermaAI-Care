import { axiosClient } from "./axiosConfig";

export interface DetectionResponse {
  id: string;
  imageUrl: string;
  imageWithBoxesUrl: string | null;
  message: string;
  detections: {
    bbox: number[];
    detection_confidence: number;
    class: string;
    class_confidence: number;
    explanation?: {
      name: string;
      description: string;
      isCancerous: boolean | null;
      severity: string;
      recommendation: string;
    };
  }[];
}

// Create a function to upload and analyze an image
export const analyzeImage = async (file: File): Promise<DetectionResponse> => {
  const formData = new FormData();
  formData.append("image", file);

  const response = await axiosClient.post<DetectionResponse>(
    "/prediction/detect",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};

export const predictionApi = {
  getUserPredictionHistory: async () => {
    try {
      const response = await axiosClient.get("/prediction/history");
      return response.data;
    } catch (error) {
      console.error("Error fetching prediction history:", error);
      throw error;
    }
  },
  
  getDoctorPatientPredictionHistory: async (patientId: string) => {
    try {
      const response = await axiosClient.get(`/prediction/doctor/history/${patientId}`);
      return response.data;
    } catch (error) {
      console.error("Error fetching patient prediction history:", error);
      throw error;
    }
  }
};
