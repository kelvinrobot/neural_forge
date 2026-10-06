import type { Report } from "../../api/types";
import { Icon } from "../ui/Icon";

export function UrgentAlert({ report, onSelectFinding }: { report: Report; onSelectFinding: (code: string) => void }) {
  const urgent = report.findings.filter((finding) => finding.red_flag && finding.status === "abnormal");
  if (urgent.length === 0) return null;
  const firstRecommendation = report.interpretation.recommendations[0];
  return (
    <section className="rounded-2xl border-2 border-red-500 bg-red-50 p-5 text-red-900 dark:bg-red-950 dark:text-red-100 print:break-inside-avoid" role="alert">
      <div className="flex items-center gap-2 text-lg font-bold">
        <Icon name="alert" size={24} />
        Urgent attention required
      </div>
      <ul className="mt-3 space-y-1 text-sm">
        {urgent.map((finding) => (
          <li key={finding.code}>
            <button type="button" className="font-semibold underline underline-offset-2" onClick={() => onSelectFinding(finding.code)}>
              {finding.title}
            </button>
          </li>
        ))}
      </ul>
      {firstRecommendation && <p className="mt-3 text-sm font-medium">{firstRecommendation}</p>}
    </section>
  );
}
