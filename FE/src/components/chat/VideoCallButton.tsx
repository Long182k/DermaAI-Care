import { VideoIcon } from "lucide-react"
import { Button } from "@/components/ui/button"

interface VideoCallButtonProps {
  onStartCall: () => void
}

export default function VideoCallButton({ onStartCall }: VideoCallButtonProps) {
  return (
    <Button variant="ghost" size="icon" onClick={onStartCall} className="text-gray-600 hover:bg-gray-100 rounded-full">
      <VideoIcon className="h-5 w-5" />
    </Button>
  )
}
