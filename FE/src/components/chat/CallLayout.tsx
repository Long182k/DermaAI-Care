import {
  StreamTheme,
  SpeakerLayout,
  CallControls,
  useCall,
  CallingState,
} from '@stream-io/video-react-sdk';
import '@stream-io/video-react-sdk/dist/css/styles.css';
type TProps = {
  handleClose: () => void;
}
export default function CallLayout({handleClose}: TProps): JSX.Element {

  return (
    <StreamTheme>
      <SpeakerLayout participantsBarPosition='bottom' />
      <CallControls onLeave={handleClose}/>
    </StreamTheme>
  );
}
