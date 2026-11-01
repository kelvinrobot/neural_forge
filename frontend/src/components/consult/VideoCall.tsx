import { useRef } from "react";
import type { CallToken } from "../../api/types";
import { useAgoraCall, type ConnectionQuality } from "../../hooks/useAgoraCall";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";
import { Spinner } from "../ui/Spinner";

const qualityLabel: Record<ConnectionQuality, { text: string; dot: string }> = {
  unknown: { text: "Checking connection", dot: "bg-ink-400" },
  good: { text: "Good connection", dot: "bg-emerald-500" },
  fair: { text: "Fair connection", dot: "bg-amber-400" },
  poor: { text: "Poor connection", dot: "bg-orange-500" },
  down: { text: "Connection lost", dot: "bg-red-600" },
};

interface VideoCallProps {
  fetchToken: () => Promise<CallToken>;
  remoteName: string;
  onEnd: () => void;
  ending: boolean;
}

export function VideoCall({ fetchToken, remoteName, onEnd, ending }: VideoCallProps) {
  const localRef = useRef<HTMLDivElement>(null);
  const remoteRef = useRef<HTMLDivElement>(null);
  const call = useAgoraCall({ fetchToken, localRef, remoteRef });
  const quality = qualityLabel[call.quality];

  return (
    <div className="space-y-3">
      <div className="relative aspect-[3/4] w-full overflow-hidden rounded-2xl bg-ink-950 sm:aspect-video">
        <div ref={remoteRef} className="absolute inset-0" />
        {!call.remoteVideo && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-ink-200">
            {call.state === "joining" ? (
              <>
                <Spinner size={32} label="Connecting" />
                <p className="text-sm">Connecting to the call…</p>
              </>
            ) : call.state === "token_error" ? (
              <>
                <Icon name="alert" size={32} className="text-red-400" />
                <p className="text-sm font-semibold">The video call could not start</p>
                <Button variant="secondary" size="sm" icon="repeat" onClick={call.retry}>
                  Retry
                </Button>
              </>
            ) : call.state === "error" ? (
              <>
                <Icon name="alert" size={32} className="text-red-400" />
                <p className="max-w-sm text-sm">{call.error}</p>
              </>
            ) : (
              <p className="text-sm">{call.remotePresent ? `${remoteName} has turned the camera off` : `Waiting for ${remoteName} to join…`}</p>
            )}
          </div>
        )}
        <div className="absolute bottom-3 right-3 aspect-[3/4] w-24 overflow-hidden rounded-[8px] border-2 border-white/70 bg-ink-800 shadow-lg sm:w-36">
          <div ref={localRef} className="h-full w-full" />
          {!call.cameraOn && (
            <div className="absolute inset-0 flex items-center justify-center bg-ink-800 text-ink-300">
              <Icon name="videoOff" />
            </div>
          )}
        </div>
        {call.state === "joined" && (
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1 text-xs text-white">
            <span className={`h-2 w-2 rounded-full ${quality.dot}`} />
            {quality.text}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button variant="secondary" icon={call.micOn ? "mic" : "micOff"} onClick={call.toggleMic} disabled={call.state !== "joined"} aria-pressed={!call.micOn}>
          {call.micOn ? "Mute" : "Unmute"}
        </Button>
        <Button variant="secondary" icon={call.cameraOn ? "video" : "videoOff"} onClick={call.toggleCamera} disabled={call.state !== "joined"} aria-pressed={!call.cameraOn}>
          {call.cameraOn ? "Camera off" : "Camera on"}
        </Button>
        <Button variant="danger" icon="phoneOff" onClick={onEnd} disabled={ending}>
          End call
        </Button>
      </div>
    </div>
  );
}
