import { analyzeImage, DetectionResponse } from "@/api/detection";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useMutation } from "@tanstack/react-query";
import {
  AlertTriangle,
  Brain,
  Calendar,
  CheckCircle,
  Image,
  Info,
  Loader2,
  ScanLine,
} from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackButton } from "./common/BackButton";

interface Detection {
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
}

export const AIAnalysis = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] =
    useState<DetectionResponse | null>(null);

  // Use React Query's useMutation for the API call
  const { mutate, isPending } = useMutation({
    mutationFn: analyzeImage,
    onSuccess: (data) => {
      setAnalysisResult(data);
    },
    onError: (error) => {
      console.error("Error analyzing image:", error);
    },
  });

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
      // Reset analysis result when a new image is uploaded
      setAnalysisResult(null);
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      mutate(selectedFile);
    }
  };

  const navigate = useNavigate();

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "high":
        return "bg-red-100 text-red-800";
      case "moderate":
      case "moderate to high":
        return "bg-orange-100 text-orange-800";
      case "low":
      case "low to moderate":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getCancerousStatus = (isCancerous: boolean | null) => {
    if (isCancerous === true) {
      return (
        <div className="flex items-center text-red-600">
          <AlertTriangle className="h-4 w-4 mr-1" />
          <span className="font-medium">Cancerous</span>
        </div>
      );
    } else if (isCancerous === false) {
      return (
        <div className="flex items-center text-green-600">
          <CheckCircle className="h-4 w-4 mr-1" />
          <span className="font-medium">Non-cancerous</span>
        </div>
      );
    } else {
      return (
        <div className="flex items-center text-gray-600">
          <Info className="h-4 w-4 mr-1" />
          <span className="font-medium">Unknown</span>
        </div>
      );
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <BackButton route="services" />
      <div className="flex justify-end mb-4">
        <Button
          variant="outline"
          onClick={() => navigate("/prediction-history")}
        >
          Prediction History
        </Button>
      </div>
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary" />
            AI Skin Cancer Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="relative">
                <Input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  id="image-upload"
                />
                <label
                  htmlFor="image-upload"
                  className={cn(
                    "flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer",
                    "hover:bg-muted/50 transition-colors",
                    selectedImage
                      ? "border-primary"
                      : "border-muted-foreground/25"
                  )}
                >
                  {selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Selected"
                      className="h-full w-full object-cover rounded-lg"
                    />
                  ) : (
                    <div className="flex flex-col items-center gap-2 text-muted-foreground">
                      <Image className="h-12 w-12" />
                      <p>Click or drag image to upload</p>
                      <p className="text-sm">Supported formats: JPG, PNG</p>
                    </div>
                  )}
                </label>
              </div>
              <Button
                onClick={handleAnalyze}
                disabled={!selectedImage || isPending}
                className="w-full"
              >
                {isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <ScanLine className="mr-2 h-4 w-4" />
                    Analyze Image
                  </>
                )}
              </Button>
            </div>
            <Card className="bg-muted/50">
              <CardHeader>
                <CardTitle className="text-lg">Analysis Results</CardTitle>
              </CardHeader>
              <CardContent>
                {isPending ? (
                  <div className="flex items-center justify-center h-48">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : analysisResult ? (
                  <div className="space-y-4">
                    {analysisResult.imageWithBoxesUrl ? (
                      <div className="rounded-lg overflow-hidden">
                        <img
                          src={analysisResult.imageWithBoxesUrl}
                          alt="Analysis Result"
                          className="w-full h-auto"
                        />
                      </div>
                    ) : null}

                    <div className="space-y-2">
                      <h3 className="font-medium">Detected Lesions:</h3>
                      {analysisResult.detections.length > 0 ? (
                        <Accordion type="single" collapsible className="w-full">
                          {analysisResult.detections.map(
                            (detection: Detection, index) => (
                              <AccordionItem
                                key={index}
                                value={`item-${index}`}
                              >
                                <AccordionTrigger className="py-2 px-4 bg-background rounded-md hover:bg-muted">
                                  <div className="flex justify-between items-center w-full">
                                    <div className="flex items-center gap-2">
                                      <Badge
                                        variant="outline"
                                        className="font-semibold"
                                      >
                                        {detection.explanation?.name ||
                                          detection.class}
                                      </Badge>
                                      {detection.explanation &&
                                        getCancerousStatus(
                                          detection.explanation.isCancerous
                                        )}
                                    </div>
                                    <span className="text-sm">
                                      Confidence:{" "}
                                      {(
                                        detection.detection_confidence * 100
                                      ).toFixed(1)}
                                      %
                                    </span>
                                  </div>
                                </AccordionTrigger>
                                <AccordionContent>
                                  {detection.explanation ? (
                                    <div className="p-4 bg-white rounded-md mt-2 space-y-3">
                                      <div>
                                        <h4 className="font-medium text-sm text-muted-foreground">
                                          What is it?
                                        </h4>
                                        <p>
                                          {detection.explanation.description}
                                        </p>
                                      </div>

                                      <div className="flex flex-wrap gap-2">
                                        <div className="flex-1 min-w-[120px]">
                                          <h4 className="font-medium text-sm text-muted-foreground">
                                            Status
                                          </h4>
                                          {getCancerousStatus(
                                            detection.explanation.isCancerous
                                          )}
                                        </div>
                                      </div>

                                      <div>
                                        <h4 className="font-medium text-sm text-muted-foreground">
                                          Recommendation
                                        </h4>
                                        <p>
                                          {detection.explanation.recommendation}
                                        </p>
                                      </div>

                                      <div className="pt-2">
                                        <Button
                                          onClick={() => navigate("/booking")}
                                          className="w-full"
                                        >
                                          <Calendar className="mr-2 h-4 w-4" />
                                          Book Appointment
                                        </Button>
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="p-4 bg-white rounded-md mt-2">
                                      <p className="text-muted-foreground">
                                        No detailed information available for
                                        this condition.
                                      </p>
                                      <div className="pt-4">
                                        <Button
                                          onClick={() => navigate("/booking")}
                                          className="w-full"
                                        >
                                          <Calendar className="mr-2 h-4 w-4" />
                                          Book Appointment
                                        </Button>
                                      </div>
                                    </div>
                                  )}
                                </AccordionContent>
                              </AccordionItem>
                            )
                          )}
                        </Accordion>
                      ) : (
                        <p className="text-muted-foreground">
                          No lesions detected
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-muted-foreground">
                    <p>Upload an image to see AI analysis results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
