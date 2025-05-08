import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ArrowLeft,
  Brain,
  Image,
  Loader2,
  ScanLine,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { BackButton } from "./common/BackButton";
import { useMutation } from "@tanstack/react-query";
import { analyzeImage, DetectionResponse } from "@/api/detection";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";

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
  const navigate = useNavigate(); // <-- Add this line

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
                        <ul className="space-y-2">
                          {analysisResult.detections.map((detection, index) => (
                            <li
                              key={index}
                              className="p-2 bg-background rounded-md"
                            >
                              <div className="flex justify-between items-center">
                                <Badge
                                  variant="outline"
                                  className="font-semibold"
                                >
                                  {detection.class}
                                </Badge>
                                <span className="text-sm">
                                  Confidence:{" "}
                                  {/* {(detection.class_confidence * 100).toFixed(
                                    1
                                  )} */}
                                  {(
                                    detection.detection_confidence * 100
                                  ).toFixed(1)}
                                  %
                                </span>
                              </div>
                            </li>
                          ))}
                        </ul>
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
