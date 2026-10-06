import type { ReportQuality, ReportStep } from "../../api/types";
import { formatValue, titleCase } from "../../utils/format";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";

function statusClass(status: string): string {
  if (status === "completed" || status === "complete") return "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200";
  if (status === "skipped") return "bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200";
  return "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200";
}

export function QualitySection({ quality, steps }: { quality: ReportQuality; steps: ReportStep[] }) {
  const stats = [
    { label: "Completed", value: `${quality.completed_steps} / ${quality.total_steps}` },
    { label: "Skipped", value: String(quality.skipped_steps) },
    { label: "Incomplete", value: String(quality.incomplete_steps) },
  ];
  return (
    <Card title="Examination quality">
      <dl className="grid grid-cols-3 gap-3">
        {stats.map((stat) => (
          <div key={stat.label} className="rounded-xl bg-ink-50 p-3 text-center dark:bg-ink-800">
            <dt className="text-xs text-ink-500 dark:text-ink-400">{stat.label}</dt>
            <dd className="text-lg font-semibold text-ink-900 dark:text-ink-50">{stat.value}</dd>
          </div>
        ))}
      </dl>
      {quality.notes.length > 0 && (
        <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-ink-600 dark:text-ink-300">
          {quality.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      <ul className="mt-4 divide-y divide-ink-100 text-sm dark:divide-ink-800">
        {steps.map((step) => (
          <li key={step.step_id} className="flex flex-wrap items-center gap-2 py-2">
            <span className="flex-1 text-ink-800 dark:text-ink-100">{step.title}</span>
            <span className="text-xs text-ink-500 dark:text-ink-400">
              {formatValue(step.duration_seconds, "s")} · {step.attempts} {step.attempts === 1 ? "attempt" : "attempts"}
            </span>
            <Badge className={statusClass(step.status)}>{titleCase(step.status)}</Badge>
            {step.note && <p className="w-full text-xs text-ink-500 dark:text-ink-400">{step.note}</p>}
          </li>
        ))}
      </ul>
    </Card>
  );
}
