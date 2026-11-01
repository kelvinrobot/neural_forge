import type { Report } from "../../api/types";
import { titleCase } from "../../utils/format";
import { findingStatusLabel, findingStatusTone, likelihoodTone, overallStatusTone, urgencyTone } from "../../utils/statusStyles";
import { EvidenceImage } from "../report/EvidenceImage";
import { MeasurementTable } from "../report/MeasurementTable";
import { Badge } from "../ui/Badge";

export function ReportSummaryPanel({ report }: { report: Report }) {
  const tone = overallStatusTone[report.overall_status] ?? overallStatusTone.inconclusive;
  const urgency = urgencyTone[report.referral_urgency] ?? urgencyTone.none;
  const flagged = report.findings.filter((finding) => finding.status === "abnormal" || finding.status === "borderline");
  return (
    <aside className="space-y-4 rounded-2xl border border-ink-100 bg-white p-4 dark:border-ink-800 dark:bg-ink-900">
      <div>
        <p className="text-xs font-medium text-brand-700 dark:text-brand-300">Shared report</p>
        <h2 className="font-semibold text-ink-900 dark:text-ink-50">{report.exam_title}</h2>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Badge className={tone.badge}>{report.overall_status_label || tone.label}</Badge>
          <Badge className={urgency.badge}>{urgency.label}</Badge>
        </div>
      </div>
      {report.interpretation.summary && (
        <section>
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">Clinician summary</h3>
          <p className="mt-1 whitespace-pre-line text-sm leading-relaxed text-ink-700 dark:text-ink-200">{report.interpretation.summary}</p>
        </section>
      )}
      {report.interpretation.possible_conditions.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">Possible conditions</h3>
          <ul className="mt-1 space-y-1">
            {report.interpretation.possible_conditions.map((condition) => (
              <li key={condition.name} className="flex flex-wrap items-center justify-between gap-2 text-sm text-ink-700 dark:text-ink-200">
                {condition.name}
                <Badge className={likelihoodTone(condition.likelihood)}>{titleCase(condition.likelihood)}</Badge>
              </li>
            ))}
          </ul>
        </section>
      )}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">Abnormal and borderline findings</h3>
        {flagged.length === 0 && <p className="text-sm text-ink-500 dark:text-ink-400">None.</p>}
        {flagged.map((finding) => (
          <article key={finding.code} className="space-y-2 rounded-xl bg-ink-50 p-3 dark:bg-ink-800">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-semibold text-ink-900 dark:text-ink-50">{finding.title}</span>
              <Badge className={findingStatusTone[finding.status].badge}>{findingStatusLabel[finding.status]}</Badge>
            </div>
            <MeasurementTable measurements={finding.measurements} />
            {finding.evidence_image && <EvidenceImage image={finding.evidence_image} title={finding.title} />}
          </article>
        ))}
      </section>
    </aside>
  );
}
