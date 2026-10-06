import type { Report } from "../../api/types";
import { formatDateTime, titleCase } from "../../utils/format";
import { overallStatusTone, urgencyTone } from "../../utils/statusStyles";
import { Badge } from "../ui/Badge";

export function StatusBanner({ report }: { report: Report }) {
  const tone = overallStatusTone[report.overall_status] ?? overallStatusTone.inconclusive;
  const urgency = urgencyTone[report.referral_urgency] ?? urgencyTone.none;
  const { patient } = report;
  return (
    <header className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-brand-700 dark:text-brand-300">Examination report</p>
          <h1 className="text-2xl font-bold text-ink-900 dark:text-ink-50 sm:text-3xl">{report.exam_title}</h1>
          <p className="mt-1 text-sm text-ink-500 dark:text-ink-400">
            {formatDateTime(report.generated_at)} · Age {patient.age ?? "not given"} · {titleCase(patient.sex)}
          </p>
        </div>
        <Badge className={urgency.badge}>{urgency.label}</Badge>
      </div>
      <div className={`flex flex-wrap items-center justify-between gap-2 rounded-2xl px-5 py-4 ${tone.banner}`} role="status">
        <span className="text-lg font-semibold">{report.overall_status_label || tone.label}</span>
        {report.stopped_early && <span className="text-sm opacity-90">Examination ended early</span>}
      </div>
    </header>
  );
}
