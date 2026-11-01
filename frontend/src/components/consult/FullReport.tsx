import type { Report } from "../../api/types";
import { findingStatusLabel, findingStatusOrder } from "../../utils/statusStyles";
import { ConditionCard } from "../report/ConditionCard";
import { FindingCard, findingAnchor } from "../report/FindingCard";
import { QualitySection } from "../report/QualitySection";
import { Recommendations } from "../report/Recommendations";
import { StatusBanner } from "../report/StatusBanner";
import { SummaryCard } from "../report/SummaryCard";

function scrollToFinding(code: string): void {
  document.getElementById(findingAnchor(code))?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function FullReport({ report }: { report: Report }) {
  const grouped = findingStatusOrder
    .map((status) => ({ status, findings: report.findings.filter((finding) => finding.status === status) }))
    .filter((group) => group.findings.length > 0);
  return (
    <div className="space-y-5">
      <StatusBanner report={report} />
      <SummaryCard interpretation={report.interpretation} onReadAloud={() => undefined} readAloudAvailable={false} />
      {report.interpretation.possible_conditions.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold text-ink-900 dark:text-ink-50">Possible conditions</h2>
          {report.interpretation.possible_conditions.map((condition) => (
            <ConditionCard key={condition.name} condition={condition} findings={report.findings} onSelectFinding={scrollToFinding} />
          ))}
        </section>
      )}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-ink-900 dark:text-ink-50">Findings</h2>
        {grouped.length === 0 && <p className="text-sm text-ink-500">No findings were reported.</p>}
        {grouped.map((group) => (
          <div key={group.status} className="space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
              {findingStatusLabel[group.status]} ({group.findings.length})
            </h3>
            {group.findings.map((finding) => (
              <FindingCard key={finding.code} finding={finding} highlighted={false} />
            ))}
          </div>
        ))}
      </section>
      <Recommendations items={report.interpretation.recommendations} />
      <QualitySection quality={report.quality} steps={report.steps} />
    </div>
  );
}
