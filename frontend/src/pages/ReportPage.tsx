import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { Finding, Report } from "../api/types";
import { ConditionCard } from "../components/report/ConditionCard";
import { ConsultPrompt } from "../components/report/ConsultPrompt";
import { FindingCard, findingAnchor } from "../components/report/FindingCard";
import { QualitySection } from "../components/report/QualitySection";
import { QuestionChat } from "../components/report/QuestionChat";
import { Recommendations } from "../components/report/Recommendations";
import { ReportActions } from "../components/report/ReportActions";
import { StatusBanner } from "../components/report/StatusBanner";
import { SummaryCard } from "../components/report/SummaryCard";
import { UrgentAlert } from "../components/report/UrgentAlert";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Spinner } from "../components/ui/Spinner";
import { useSpeech } from "../hooks/useSpeech";
import { useExamStore } from "../state/examStore";
import { findingStatusLabel, findingStatusOrder } from "../utils/statusStyles";

const REPORT_POLL_MS = 3000;

export function ReportPage() {
  const { sessionId = "" } = useParams();
  const navigate = useNavigate();
  const storedReport = useExamStore((state) => state.report);
  const setReport = useExamStore((state) => state.setReport);
  const [report, setLocalReport] = useState<Report | null>(storedReport?.session_id === sessionId ? storedReport : null);
  const [failure, setFailure] = useState<string | null>(null);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const { speak, supported } = useSpeech();

  useEffect(() => {
    if (report) return undefined;
    let cancelled = false;
    let timer: number | null = null;
    const load = async () => {
      try {
        const fetched = await api.report(sessionId);
        if (cancelled) return;
        setLocalReport(fetched);
        setReport(fetched);
      } catch (error) {
        if (cancelled) return;
        if (error instanceof ApiError && error.status === 409) {
          timer = window.setTimeout(load, REPORT_POLL_MS);
        } else {
          setFailure(error instanceof ApiError && error.status === 404 ? "This report no longer exists." : "The report could not be loaded.");
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [report, sessionId, setReport]);

  const selectFinding = useCallback((code: string) => {
    const element = document.getElementById(findingAnchor(code));
    if (!element) return;
    element.scrollIntoView({ behavior: "smooth", block: "start" });
    element.focus({ preventScroll: true });
    setHighlighted(code);
    window.setTimeout(() => setHighlighted((current) => (current === code ? null : current)), 2500);
  }, []);

  if (failure) {
    return (
      <div className="flex flex-col items-center gap-4 py-20 text-center">
        <p className="text-lg">{failure}</p>
        <Button onClick={() => navigate("/")}>Back to examinations</Button>
      </div>
    );
  }
  if (!report) {
    return (
      <div className="flex flex-col items-center gap-3 py-20 text-brand-600">
        <Spinner size={36} label="Loading report" />
        <p className="text-sm text-ink-500">Preparing the report…</p>
      </div>
    );
  }

  const grouped = findingStatusOrder
    .map((status) => ({ status, findings: report.findings.filter((finding: Finding) => finding.status === status) }))
    .filter((group) => group.findings.length > 0);

  return (
    <article className="mx-auto max-w-5xl space-y-6">
      <UrgentAlert report={report} onSelectFinding={selectFinding} />
      <StatusBanner report={report} />
      <ConsultPrompt report={report} />
      <ReportActions report={report} />
      <SummaryCard interpretation={report.interpretation} onReadAloud={() => speak(report.speech_summary, { interrupt: true })} readAloudAvailable={supported && Boolean(report.speech_summary)} />

      {report.interpretation.possible_conditions.length > 0 && (
        <section aria-labelledby="conditions-heading">
          <h2 id="conditions-heading" className="mb-3 text-lg font-semibold text-ink-900 dark:text-ink-50">
            Possible conditions
          </h2>
          <div className="grid gap-3 md:grid-cols-2">
            {report.interpretation.possible_conditions.map((condition) => (
              <ConditionCard key={condition.name} condition={condition} findings={report.findings} onSelectFinding={selectFinding} />
            ))}
          </div>
        </section>
      )}

      <section aria-labelledby="findings-heading" className="space-y-5">
        <h2 id="findings-heading" className="text-lg font-semibold text-ink-900 dark:text-ink-50">
          Findings
        </h2>
        {grouped.length === 0 && <p className="text-sm text-ink-500">No findings were reported.</p>}
        {grouped.map((group) => (
          <div key={group.status} className="space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
              {findingStatusLabel[group.status]} ({group.findings.length})
            </h3>
            {group.findings.map((finding) => (
              <FindingCard key={finding.code} finding={finding} highlighted={highlighted === finding.code} />
            ))}
          </div>
        ))}
      </section>

      <Recommendations items={report.interpretation.recommendations} />
      <QualitySection quality={report.quality} steps={report.steps} />
      <QuestionChat sessionId={report.session_id} onSpeak={(text) => speak(text, { interrupt: true })} />

      <Card className="text-xs leading-relaxed text-ink-600 dark:text-ink-300">
        <p className="whitespace-pre-line">{report.disclaimer}</p>
      </Card>
    </article>
  );
}
