import type { Step } from "../../api/types";
import type { StepRunStatus } from "../../state/examStore";
import { Icon, type IconName } from "../ui/Icon";

const STATUS: Record<StepRunStatus, { icon: IconName; className: string; label: string }> = {
  pending: { icon: "dot", className: "text-ink-300 dark:text-ink-600", label: "Pending" },
  active: { icon: "chevronRight", className: "text-brand-600 dark:text-brand-300", label: "In progress" },
  completed: { icon: "check", className: "text-emerald-600 dark:text-emerald-400", label: "Completed" },
  skipped: { icon: "skip", className: "text-ink-400", label: "Skipped" },
  incomplete: { icon: "alert", className: "text-amber-600 dark:text-amber-400", label: "Incomplete" },
  repeated: { icon: "repeat", className: "text-brand-600 dark:text-brand-300", label: "Repeated" },
};

interface StepTimelineProps {
  steps: Step[];
  status: Record<string, StepRunStatus>;
  currentStepId: string | null;
}

export function StepTimeline({ steps, status, currentStepId }: StepTimelineProps) {
  return (
    <ol className="space-y-1" aria-label="Examination steps">
      {steps.map((step) => {
        const state = status[step.id] ?? "pending";
        const style = STATUS[state];
        const current = step.id === currentStepId;
        return (
          <li
            key={step.id}
            className={`flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm ${current ? "bg-brand-50 font-semibold text-ink-900 dark:bg-brand-950 dark:text-ink-50" : "text-ink-600 dark:text-ink-300"}`}
            aria-current={current ? "step" : undefined}
          >
            <span className={style.className}>
              <Icon name={style.icon} size={18} />
            </span>
            <span className="flex-1">
              {step.number}. {step.title}
            </span>
            <span className="sr-only">{style.label}</span>
          </li>
        );
      })}
    </ol>
  );
}
