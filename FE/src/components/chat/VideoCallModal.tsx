import { useEffect, useState } from "react"
import { type Call, CallingState, StreamCall, useCallStateHooks } from "@stream-io/video-react-sdk"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import CallLayout from "./CallLayout"

interface VideoCallModalProps {
  call: Call
  onEndCall: () => void
}

export default function VideoCallModal({ call, onEndCall }: VideoCallModalProps) {
  const [isOpen, setIsOpen] = useState(true)
  const { useCallCallingState } = useCallStateHooks()
const callingState = useCallCallingState()

  useEffect(() => {
    // Join the call when the component mounts
    const joinCall = async () => {
      try {
        if (callingState !== CallingState.JOINED) {
          await call.join()
          await call.camera.enable()
          await call.microphone.enable()
        }
      } catch (error) {
        console.error("Error joining call:", error)
      }
    }

    joinCall()
    setIsOpen(true)

    // Clean up when the component unmounts
    return () => {
      if (callingState === CallingState.JOINED) {
        call.leave()
      }
    }
  }, [call, callingState])

  const handleClose = async () => {
    onEndCall()
    setIsOpen(false)
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      if (!open) handleClose()
      else setIsOpen(true)
    }}>
      <DialogContent 
        className="sm:max-w-[90vw] max-h-[90vh] w-fit h-fit p-0 overflow-hidden"
        onEscapeKeyDown={(e) => {e.preventDefault()}}
        onInteractOutside={(e) =>{e.preventDefault()}}
      >
         <StreamCall call={call} >
        <CallLayout handleClose={handleClose} />
      </StreamCall>
      </DialogContent>
    </Dialog>
  )
}
