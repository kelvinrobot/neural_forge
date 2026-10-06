import type { Finding } from "../../api/types";
import { formatPercent } from "../../utils/format";
import { findingStatusLabel, findingStatusTone } from "../../utils/statusStyles";
import { Badge } from "../ui/Badge";
import { Icon } from "../ui/Icon";
import { EvidenceImage } from "./EvidenceImage";
import { MeasurementTable } from "./MeasurementTable";

export function findingAnchor(code: string): string {
  return `finding-${code.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

export function FindingCard({ finding, highlighted }: { finding: Finding; highlighted: boolean }) {
  const tone = findingStatusTone[finding.status] ?? findingStatusTone.inconclusive;
  const confidence = Math.max(0, Math.min(1, finding.confidence <= 1 ? finding.confidence : finding.confidence / 100));
  return (
    <article
      id={findingAnchor(finding.code)}
      tabIndex={-1}
      className={`scroll-mt-24 rounded-2xl border bg-white p-5 shadow-soft outline-none transition-shadow dark:bg-ink-900 print:break-inside-avoid print:shadow-none ${highlighted ? "border-brand-500 ring-2 ring-brand-300" : "border-ink-100 dark:border-ink-800"}`}
    >
      <header className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="flex items-center gap-2 font-semibold text-ink-900 dark:text-ink-50">
          {finding.red_flag && (
            <span className="text-red-600">
              <Icon name="alert" />
              <span className="sr-only">Red flag:</span>
            </span>
          )}
          {finding.title}
        </h3>
        <Badge className={tone.badge}>{findingStatusLabel[finding.status] ?? finding.status}</Badge>
      </header>
      <div className="mt-3 flex items-center gap-3">
        <span className="text-xs font-medium text-ink-500 dark:text-ink-400">Confidence</span>
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-100 dark:bg-ink-800">
          <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${confidence * 100}%` }} />
        </div>
        <span className="text-xs font-semibold tabular-nums text-ink-700 dark:text-ink-200">{formatPercent(confidence)}</span>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-ink-700 dark:text-ink-200">{finding.explanation}</p>
      <div className={`mt-4 grid gap-4 ${finding.evidence_image ? "md:grid-cols-[1fr_200px]" : ""}`}>
        <MeasurementTable measurements={finding.measurements} />
        {finding.evidence_image && <EvidenceImage image={finding.evidence_image} title={finding.title} />}
      </div>
    </article>
  );
}
