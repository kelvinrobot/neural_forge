import { useNavigate } from "react-router-dom";
import type { Report } from "../../api/types";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";

function needsEmergencyCare(report: Report): boolean {
  return (
    report.referral_urgency === "urgent" ||
    report.interpretation.referral_urgency === "urgent" ||
    report.findings.some((finding) => finding.red_flag && finding.status === "abnormal")
  );
}

export function ConsultPrompt({ report }: { report: Report }) {
  const navigate = useNavigate();
  return (
    <section className="space-y-3 rounded-2xl border border-brand-200 bg-brand-50 p-5 dark:border-brand-900 dark:bg-brand-950 print:hidden">
      {needsEmergencyCare(report) && (
        <p role="alert" className="flex gap-2 rounded-xl bg-red-600 p-4 text-sm font-semibold text-white">
          <Icon name="alert" size={20} className="shrink-0" />
          This result may need emergency care. Go to the nearest emergency department or call emergency services now.
        </p>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-ink-700 dark:text-ink-200">Talk through this report with a doctor on a video call.</p>
        <Button size="lg" icon="video" onClick={() => navigate(`/consult/${encodeURIComponent(report.session_id)}`)}>
          Consult a doctor
        </Button>
      </div>
    </section>
  );
}
