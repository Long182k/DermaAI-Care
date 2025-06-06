import React, { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileImage, Loader2 } from "lucide-react";
import { format, isToday, isThisWeek, isThisMonth } from "date-fns";
import { useNavigate, useParams } from "react-router-dom";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useQuery } from "@tanstack/react-query";
import { predictionApi } from "@/api/detection";
import { Navbar } from "@/components/Navbar";
import type { DetectionResponse } from "@/api/detection";
import { useAppStore } from "@/store";

// Define filter types
type FilterType = "day" | "week" | "month" | "all";

// Define Prediction type
interface Prediction {
  id: string;
  imageUrl: string;
  imageWithBoxesUrl: string | null;
  status: string;
  createdAt: string;
  result: DetectionResponse;
}

const PredictionHistoryPage = () => {
  const [filter, setFilter] = useState<FilterType>("all");
  const navigate = useNavigate();
  const { patientId } = useParams<{ patientId?: string }>();
  const { userInfo } = useAppStore();
  const isDoctor = userInfo?.role === "DOCTOR";
  const isViewingPatient = isDoctor && patientId;

  // Use React Query to fetch prediction history
  const {
    data: predictions = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["predictionHistory", patientId],
    queryFn: () => {
      if (isViewingPatient) {
        return predictionApi.getDoctorPatientPredictionHistory(patientId);
      } else {
        return predictionApi.getUserPredictionHistory();
      }
    },
  });

  // Filter predictions based on date
  const filterPredictionsByDate = (predictions: Prediction[]) => {
    return predictions.filter((prediction) => {
      const predictionDate = new Date(prediction.createdAt);

      switch (filter) {
        case "day":
          return isToday(predictionDate);
        case "week":
          return isThisWeek(predictionDate);
        case "month":
          return isThisMonth(predictionDate);
        case "all":
        default:
          return true;
      }
    });
  };

  const filteredPredictions = filterPredictionsByDate(predictions);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return format(date, "EEEE, MMMM d, yyyy");
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return format(date, "h:mm a");
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "COMPLETED":
        return "bg-green-100 text-green-800";
      case "PENDING":
        return "bg-yellow-100 text-yellow-800";
      case "FAILED":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getFilterLabel = () => {
    switch (filter) {
      case "day":
        return "Today";
      case "week":
        return "This Week";
      case "month":
        return "This Month";
      case "all":
      default:
        return "All Time";
    }
  };

  const handleViewDetails = (id: string) => {
    navigate(`/analysis/${id}`);
  };

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-3xl font-bold mb-6">
          {isViewingPatient ? "Patient's Skin Prediction History" : "My Skin Prediction History"}
        </h1>
        <div className="flex justify-center items-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-3xl font-bold mb-6">
          {isViewingPatient ? "Patient's Skin Prediction History" : "My Skin Prediction History"}
        </h1>
        <div className="flex justify-center items-center h-64">
          <p className="text-red-500">
            Failed to load prediction history. Please try again later.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-20">
      <Navbar />
      <div className="flex items-center mb-6">
        <Button
          variant="ghost"
          size="icon"
          className="mr-2"
          onClick={() => isViewingPatient ? navigate("/appointments") : navigate("/analysis")}
          aria-label="Back"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
        </Button>
        <h1 className="text-3xl font-bold">
          {isViewingPatient ? "Patient's Skin Prediction History" : "My Skin Prediction History"}
        </h1>
      </div>

      <div className="mb-6 flex justify-between items-center">
        <h2 className="text-xl font-medium"></h2>
        <div className="flex items-center gap-2">
          {!isViewingPatient && (
            <Button variant="outline" onClick={() => navigate("/analysis")}>
              New Analysis
            </Button>
          )}
          {/* Filter dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="w-[120px]">
                {getFilterLabel()}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setFilter("day")}>
                Today
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setFilter("week")}>
                This Week
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setFilter("month")}>
                This Month
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setFilter("all")}>
                All Time
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="space-y-4">
        {filteredPredictions.length === 0 ? (
          <Card>
            <CardContent className="py-10 text-center">
              <p className="text-muted-foreground">
                {isViewingPatient ? "This patient has" : "You have"} no skin analysis history
                {filter !== "all"
                  ? ` for ${getFilterLabel().toLowerCase()}`
                  : ""}
                .
              </p>
              {!isViewingPatient && (
                <Button className="mt-4" onClick={() => navigate("/analysis")}>
                  New Skin Analysis
                </Button>
              )}
            </CardContent>
          </Card>
        ) : (
          filteredPredictions.map((prediction) => (
            <Card key={prediction.id} className="overflow-hidden">
              <div className="flex flex-col md:flex-row">
                <div className="bg-primary/10 p-6 flex flex-col justify-center items-center md:w-1/4">
                  {prediction.imageUrl ? (
                    <img
                      src={prediction.imageUrl}
                      alt="Skin lesion"
                      className="w-24 h-24 object-cover rounded-md mb-2"
                    />
                  ) : (
                    <FileImage className="h-8 w-8 text-primary mb-2" />
                  )}
                  <p className="text-lg font-medium">
                    {formatDate(prediction.createdAt)}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {formatTime(prediction.createdAt)}
                  </p>
                </div>

                <CardContent className="flex-1 p-6">
                  <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
                    <div className="flex items-center gap-3 flex-wrap">
                      <h3 className="text-xl font-semibold">
                        {prediction.result?.detections?.[0]?.class
                          ? `Detected: ${prediction.result.detections[0].class}`
                          : "Skin Analysis Result"}
                      </h3>
                      {prediction.result?.detections?.[0]?.explanation
                        ?.name && (
                        <span className="text-base text-muted-foreground font-medium ml-2">
                          {prediction.result.detections[0].explanation.name}
                        </span>
                      )}
                    </div>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-medium mt-2 md:mt-0 ${getStatusColor(
                        prediction.status
                      )}`}
                    >
                      {prediction.status}
                    </span>
                  </div>
                  {/* Explanation Section */}
                  {prediction.result?.detections?.[0]?.explanation && (
                    <div className="bg-gray-50 rounded-md p-4 mt-2 space-y-2 border border-gray-100">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-1">
                          What is it?
                        </h4>
                        <p className="text-base font-normal">
                          {
                            prediction.result.detections[0].explanation
                              .description
                          }
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-4">
                        <div className="flex-1 min-w-[120px]">
                          <h4 className="font-medium text-sm text-muted-foreground mb-1">
                            Status
                          </h4>
                          <span className="text-sm">
                            {prediction.result.detections[0].explanation
                              .isCancerous === true && (
                              <span className="text-red-600 font-semibold">
                                Cancerous
                              </span>
                            )}
                            {prediction.result.detections[0].explanation
                              .isCancerous === false && (
                              <span className="text-green-600 font-semibold">
                                Non-cancerous
                              </span>
                            )}
                            {prediction.result.detections[0].explanation
                              .isCancerous === null && (
                              <span className="text-gray-600 font-semibold">
                                Unknown
                              </span>
                            )}
                          </span>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-1">
                          Recommendation
                        </h4>
                        <p className="text-base font-normal">
                          {
                            prediction.result.detections[0].explanation
                              .recommendation
                          }
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default PredictionHistoryPage;
