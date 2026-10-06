import type { Finding, PossibleCondition } from "../../api/types";
import { titleCase } from "../../utils/format";
import { likelihoodTone } from "../../utils/statusStyles";
import { Badge } from "../ui/Badge";

interface ConditionCardProps {
  condition: PossibleCondition;
  findings: Finding[];
  onSelectFinding: (code: string) => void;
}

export function ConditionCard({ condition, findings, onSelectFinding }: ConditionCardProps) {
  const linked = condition.supporting_findings.map((reference) => {
    const match = findings.find((finding) => finding.code === reference || finding.title.toLowerCase() === reference.toLowerCase());
    return { reference, finding: match };
  });
  return (
    <article className="rounded-2xl border border-ink-100 bg-white p-4 dark:border-ink-800 dark:bg-ink-900 print:break-inside-avoid">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="font-semibold text-ink-900 dark:text-ink-50">{condition.name}</h3>
        <Badge className={likelihoodTone(condition.likelihood)}>{titleCase(condition.likelihood)}</Badge>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-ink-700 dark:text-ink-200">{condition.reasoning}</p>
      {linked.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {linked.map(({ reference, finding }) =>
            finding ? (
              <button
                key={reference}
                type="button"
                onClick={() => onSelectFinding(finding.code)}
                className="rounded-full bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-800 hover:bg-brand-100 dark:bg-brand-950 dark:text-brand-200"
              >
                {finding.title}
              </button>
            ) : (
              <span key={reference} className="rounded-full bg-ink-100 px-2.5 py-1 text-xs text-ink-600 dark:bg-ink-800 dark:text-ink-300">
                {titleCase(reference)}
              </span>
            ),
          )}
        </div>
      )}
    </article>
  );
}
