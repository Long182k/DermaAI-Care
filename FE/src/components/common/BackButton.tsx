import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface BackButtonProps {
  route?: string;
}
export const BackButton = ({ route }: BackButtonProps) => {
  const navigate = useNavigate();
  return (
    <Button
      variant="ghost"
      size="icon"
      className="absolute top-4 left-4"
      onClick={() => (route ? navigate(`/${route}`) : navigate("/"))}
    >
      <ArrowLeft className="h-6 w-6" />
    </Button>
  );
};
