import { AnimatePresence, motion } from "framer-motion";
import type { Step } from "../../api/types";
import { useExamStore } from "../../state/examStore";
import { useSettingsStore } from "../../state/settingsStore";
import { formatSeconds } from "../../utils/format";
import { GuideFigure } from "../guide/GuideFigure";
import { Icon } from "../ui/Icon";
import { ProgressRing } from "../ui/ProgressRing";

interface StepPanelProps {
  examId: string;
  step: Step;
  totalSteps: number;
  finishedSteps: number;
}

export function StepPanel({ examId, step, totalSteps, finishedSteps }: StepPanelProps) {
  const phase = useExamStore((state) => state.phase);
  const progress = useExamStore((state) => state.progress);
  const displayText = useExamStore((state) => state.displayText);
  const stepNote = useExamStore((state) => state.stepNote);
  const repeatNumber = useExamStore((state) => state.repeatNumber);
  const skinTone = useSettingsStore((state) => state.guideSkinTone);
  const overall = totalSteps > 0 ? (finishedSteps / totalSteps) * 100 : 0;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
          <span>
            Step {step.number} of {totalSteps}
          </span>
          {repeatNumber > 1 && <span className="text-brand-600 dark:text-brand-300">Attempt {repeatNumber}</span>}
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-ink-100 dark:bg-ink-800" role="progressbar" aria-label="Examination progress" aria-valuenow={Math.round(overall)} aria-valuemin={0} aria-valuemax={100}>
          <motion.div className="h-full rounded-full bg-brand-500" animate={{ width: `${overall}%` }} transition={{ duration: 0.4 }} />
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={step.id}
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -16 }}
          transition={{ duration: 0.25 }}
          className="flex flex-col gap-4"
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-ink-900 dark:text-ink-50">{step.title}</h2>
              <p className="mt-1 text-sm leading-relaxed text-ink-700 dark:text-ink-200">{displayText || step.display_text}</p>
            </div>
            {phase === "collecting" && progress && (
              <ProgressRing value={progress.percent} size={78} stroke={7} label="Step progress">
                <span className="text-center leading-tight">
                  <span className="block text-sm font-bold text-ink-900 dark:text-ink-50">{Math.round(progress.percent)}%</span>
                  <span className="block text-[10px] text-ink-500">{formatSeconds(progress.remaining)}</span>
                </span>
              </ProgressRing>
            )}
          </div>

          {stepNote && (
            <p className="rounded-xl bg-brand-50 px-3 py-2 text-sm text-brand-900 dark:bg-brand-950 dark:text-brand-100" role="status">
              {stepNote}
            </p>
          )}

          <GuideFigure examId={examId} step={step} playing={phase !== "countdown"} skinTone={skinTone} />

          {step.tips.length > 0 && (
            <ul className="space-y-1.5 text-sm text-ink-600 dark:text-ink-300">
              {step.tips.map((tip) => (
                <li key={tip} className="flex gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
                  {tip}
                </li>
              ))}
            </ul>
          )}

          {step.safety_note && (
            <div className="flex gap-2.5 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2.5 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-100" role="note">
              <Icon name="shield" className="mt-0.5 shrink-0" />
              <span>{step.safety_note}</span>
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
