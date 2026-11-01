import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { Consultation, Report } from "../api/types";
import { DoctorPhoto } from "../components/consult/DoctorPhoto";
import { ReportSummaryPanel } from "../components/consult/ReportSummaryPanel";
import { VideoCall } from "../components/consult/VideoCall";
import { Button } from "../components/ui/Button";
import { Spinner } from "../components/ui/Spinner";
import { toast } from "../components/ui/Toast";
import { findDoctor } from "../data/doctors";
import { usePolling } from "../hooks/usePolling";
import { downloadReport } from "../utils/download";

const POLL_MS = 2000;

type Outcome = "patient" | "doctor" | "closed";

const outcomeText: Record<Outcome, string> = {
  patient: "You ended the consultation",
  doctor: "The doctor ended the consultation",
  closed: "This consultation has ended",
};

function CallingIndicator({ name }: { name: string }) {
  const reduceMotion = useReducedMotion();
  return (
    <p className="flex items-center gap-2 text-lg font-semibold text-ink-900 dark:text-ink-50" role="status">
      <span className="relative flex h-3 w-3">
        <motion.span
          className="absolute inset-0 rounded-full bg-emerald-400"
          animate={reduceMotion ? undefined : { scale: [1, 2.2], opacity: [0.7, 0] }}
          transition={{ duration: 1.2, repeat: Infinity }}
        />
        <span className="relative h-3 w-3 rounded-full bg-emerald-500" />
      </span>
      Calling {name}…
    </p>
  );
}

export function ConsultationPage() {
  const { consultationId = "" } = useParams();
  const navigate = useNavigate();
  const [consultation, setConsultation] = useState<Consultation | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [ending, setEnding] = useState(false);
  const lastStatus = useRef<Consultation["status"] | null>(null);
  const endingRef = useRef(false);

  usePolling(
    async () => {
      try {
        const latest = await api.consultation(consultationId);
        if (endingRef.current) return;
        if (latest.status === "ended") {
          setOutcome((current) => current ?? (lastStatus.current === "accepted" ? "doctor" : "closed"));
        }
        lastStatus.current = latest.status;
        setConsultation(latest);
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) setFailure("This consultation no longer exists.");
      }
    },
    POLL_MS,
    outcome === null && failure === null,
  );

  const sessionId = consultation?.session_id;
  useEffect(() => {
    if (!sessionId) return undefined;
    let cancelled = false;
    api
      .report(sessionId)
      .then((loaded) => {
        if (!cancelled) setReport(loaded);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const end = async (next: Outcome | "cancel") => {
    endingRef.current = true;
    setEnding(true);
    try {
      await api.endConsultation(consultationId);
    } catch {
      toast("The server could not be told the call ended.", "error");
    }
    setEnding(false);
    if (next === "cancel") {
      navigate(`/consult/${encodeURIComponent(consultation?.session_id ?? "")}`);
    } else {
      setOutcome((current) => current ?? next);
    }
  };

  if (failure) {
    return (
      <div className="flex flex-col items-center gap-4 py-20 text-center">
        <p className="text-lg">{failure}</p>
        <Button onClick={() => navigate("/")}>Back to home</Button>
      </div>
    );
  }
  if (!consultation) {
    return (
      <div className="flex justify-center py-20 text-brand-600">
        <Spinner size={36} label="Loading consultation" />
      </div>
    );
  }

  const doctor = findDoctor(consultation.professional_id);
  const doctorName = doctor?.name ?? consultation.professional_name;
  const directory = `/consult/${encodeURIComponent(consultation.session_id)}`;

  if (outcome) {
    return (
      <div className="mx-auto flex max-w-md flex-col items-center gap-4 py-16 text-center">
        {doctor && <DoctorPhoto doctor={doctor} className="w-32" />}
        <h1 className="text-2xl font-bold text-ink-900 dark:text-ink-50">Consultation ended</h1>
        <p className="text-ink-600 dark:text-ink-300">{outcomeText[outcome]}</p>
        <div className="flex flex-wrap justify-center gap-2">
          <Button icon="download" onClick={() => report && downloadReport(report)} disabled={!report}>
            Download report
          </Button>
          <Button variant="secondary" onClick={() => navigate("/")}>
            Back to home
          </Button>
        </div>
      </div>
    );
  }

  if (consultation.status === "declined" || consultation.status === "expired") {
    return (
      <div className="mx-auto flex max-w-md flex-col items-center gap-4 py-16 text-center">
        {doctor && <DoctorPhoto doctor={doctor} className="w-32" />}
        <h1 className="text-2xl font-bold text-ink-900 dark:text-ink-50">
          {consultation.status === "declined" ? `${doctorName} could not take your call` : "No answer"}
        </h1>
        <p className="text-ink-600 dark:text-ink-300">
          {consultation.status === "declined"
            ? "The doctor declined this request. You can try again later or choose another doctor."
            : "The request timed out before the doctor answered. You can try again."}
        </p>
        <Link to={directory} className="inline-flex h-11 items-center rounded-xl bg-brand-600 px-4 text-sm font-semibold text-white hover:bg-brand-700">
          Back to doctors
        </Link>
      </div>
    );
  }

  if (consultation.status === "requested") {
    return (
      <div className="mx-auto flex max-w-md flex-col items-center gap-5 py-12 text-center">
        {doctor && <DoctorPhoto doctor={doctor} className="w-44 sm:w-52" />}
        <CallingIndicator name={doctorName} />
        <p className="text-sm text-ink-500 dark:text-ink-400">Keep this page open. The call starts as soon as the doctor accepts.</p>
        <Button variant="secondary" icon="x" onClick={() => void end("cancel")} disabled={ending}>
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-2">
        <h1 className="text-lg font-semibold text-ink-900 dark:text-ink-50">Consultation with {doctorName}</h1>
        <VideoCall channel={consultation.channel_name} remoteName={doctorName} onEnd={() => void end("patient")} ending={ending} />
      </div>
      {report ? (
        <ReportSummaryPanel report={report} />
      ) : (
        <div className="flex justify-center py-10 text-brand-600">
          <Spinner label="Loading report" />
        </div>
      )}
    </div>
  );
}
