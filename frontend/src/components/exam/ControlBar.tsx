import { useState } from "react";
import type { ControlAction, Step } from "../../api/types";
import type { ExamPhase } from "../../state/examStore";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";

interface ControlBarProps {
  phase: ExamPhase;
  paused: boolean;
  step: Step | null;
  areasCaptured: number;
  onControl: (action: ControlAction) => void;
}

const ACTIVE_PHASES: ExamPhase[] = ["preview", "waiting_ready", "positioning", "countdown", "collecting", "area_complete", "between_steps"];

export function ControlBar({ phase, paused, step, areasCaptured, onControl }: ControlBarProps) {
  const [confirmStop, setConfirmStop] = useState(false);
  const running = ACTIVE_PHASES.includes(phase);
  const canSkip = Boolean(step?.skippable) && running && phase !== "area_complete";

  return (
    <div className="flex flex-col gap-3">
      {phase === "ready_to_start" && (
        <Button size="lg" icon="play" onClick={() => onControl("start")} className="w-full">
          Start examination
        </Button>
      )}
      {phase === "waiting_ready" && (
        <Button size="lg" icon="check" onClick={() => onControl("ready")} className="w-full animate-pulse motion-reduce:animate-none">
          I'm ready
        </Button>
      )}
      {phase === "area_complete" && (
        <div className="grid grid-cols-2 gap-2">
          <p className="col-span-2 text-sm text-ink-600 dark:text-ink-300">
            {areasCaptured} {areasCaptured === 1 ? "area" : "areas"} captured.
          </p>
          <Button variant="secondary" icon="plus" onClick={() => onControl("add_area")}>
            Add another area
          </Button>
          <Button icon="chevronRight" onClick={() => onControl("next")}>
            Continue
          </Button>
        </div>
      )}
      {running && (
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" icon={paused ? "play" : "pause"} onClick={() => onControl(paused ? "resume" : "pause")}>
            {paused ? "Resume" : "Pause"}
          </Button>
          {canSkip && (
            <Button variant="secondary" icon="skip" onClick={() => onControl("skip")}>
              Skip step
            </Button>
          )}
          <Button variant="ghost" icon="stop" onClick={() => setConfirmStop(true)} className="ml-auto text-red-600 dark:text-red-400">
            Stop
          </Button>
        </div>
      )}
      <Modal
        open={confirmStop}
        title="Stop the examination?"
        onClose={() => setConfirmStop(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmStop(false)}>
              Continue examination
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                setConfirmStop(false);
                onControl("stop");
              }}
            >
              Stop now
            </Button>
          </>
        }
      >
        The examination will end now. Results are prepared only from the steps already completed.
      </Modal>
    </div>
  );
}
