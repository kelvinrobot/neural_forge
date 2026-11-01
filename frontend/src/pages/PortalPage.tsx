import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { Consultation, ReferralUrgency, Report } from "../api/types";
import { DoctorPhoto } from "../components/consult/DoctorPhoto";
import { FullReport } from "../components/consult/FullReport";
import { VideoCall } from "../components/consult/VideoCall";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Icon } from "../components/ui/Icon";
import { Spinner } from "../components/ui/Spinner";
import { toast } from "../components/ui/Toast";
import { inputClass, labelClass } from "../components/ui/fieldStyles";
import { findDoctor, type Doctor } from "../data/doctors";
import { usePolling } from "../hooks/usePolling";
import { useRingtone } from "../hooks/useRingtone";
import { formatDateTime } from "../utils/format";
import { urgencyTone } from "../utils/statusStyles";

const POLL_MS = 2000;

interface PortalAuth {
  professionalId: string;
  code: string;
  doctor: Doctor | undefined;
}

function LoginForm({ onLogin, notice }: { onLogin: (auth: PortalAuth) => void; notice: string | null }) {
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setFailure(null);
    try {
      const { professional_id } = await api.portalLogin(email.trim(), code);
      onLogin({ professionalId: professional_id, code, doctor: findDoctor(professional_id) });
    } catch (error) {
      setFailure(error instanceof ApiError && error.status === 403 ? "Wrong email or access code" : "The portal could not be reached. Please try again.");
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-sm py-10">
      <h1 className="text-2xl font-bold text-ink-900 dark:text-ink-50">Doctor portal</h1>
      <p className="mt-1 text-sm text-ink-600 dark:text-ink-300">Sign in to receive consultation requests.</p>
      <form onSubmit={submit} className="mt-6 space-y-4">
        <div>
          <label htmlFor="portal-email" className={labelClass}>
            Email
          </label>
          <input id="portal-email" type="email" autoComplete="username" required value={email} onChange={(event) => setEmail(event.target.value)} className={inputClass} />
        </div>
        <div>
          <label htmlFor="portal-code" className={labelClass}>
            Access code
          </label>
          <input id="portal-code" type="password" autoComplete="current-password" required value={code} onChange={(event) => setCode(event.target.value)} className={inputClass} />
        </div>
        {(failure ?? notice) && (
          <p role="alert" className="rounded-xl bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            {failure ?? notice}
          </p>
        )}
        <Button type="submit" className="w-full" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </Button>
      </form>
    </div>
  );
}

function RequestCard({ request, onAccept, onDecline, busy }: { request: Consultation; onAccept: () => void; onDecline: () => void; busy: boolean }) {
  const urgency = urgencyTone[request.referral_urgency as ReferralUrgency] ?? urgencyTone.none;
  return (
    <article className="rounded-2xl border-2 border-brand-500 bg-white p-5 shadow-soft dark:bg-ink-900">
      <p className="flex items-center gap-2 text-sm font-semibold text-brand-700 dark:text-brand-300">
        <Icon name="bell" size={18} className="animate-pulse motion-reduce:animate-none" />
        New consultation request
      </p>
      <h2 className="mt-2 text-2xl font-bold text-ink-900 dark:text-ink-50">{request.patient_display_name}</h2>
      <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-ink-500 dark:text-ink-400">Examination</dt>
          <dd className="font-medium text-ink-900 dark:text-ink-50">{request.exam_title}</dd>
        </div>
        <div>
          <dt className="text-ink-500 dark:text-ink-400">Overall status</dt>
          <dd className="font-medium text-ink-900 dark:text-ink-50">{request.overall_status_label || "–"}</dd>
        </div>
        <div>
          <dt className="text-ink-500 dark:text-ink-400">Referral urgency</dt>
          <dd>
            <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${urgency.badge}`}>{urgency.label}</span>
          </dd>
        </div>
        <div>
          <dt className="text-ink-500 dark:text-ink-400">Received</dt>
          <dd className="font-medium text-ink-900 dark:text-ink-50">{formatDateTime(new Date(request.created_at * 1000).toISOString())}</dd>
        </div>
      </dl>
      {request.patient_note && <p className="mt-3 rounded-xl bg-ink-50 p-3 text-sm text-ink-700 dark:bg-ink-800 dark:text-ink-200">“{request.patient_note}”</p>}
      <div className="mt-4 flex flex-wrap gap-2">
        <Button size="lg" icon="video" onClick={onAccept} disabled={busy} className="flex-1 sm:flex-none">
          Accept
        </Button>
        <Button size="lg" variant="secondary" icon="x" onClick={onDecline} disabled={busy} className="flex-1 sm:flex-none">
          Decline
        </Button>
      </div>
    </article>
  );
}

function CallView({ consultation, onClose }: { consultation: Consultation; onClose: (notice: string | null) => void }) {
  const [report, setReport] = useState<Report | null>(null);
  const [reportFailed, setReportFailed] = useState(false);
  const [tab, setTab] = useState<"video" | "report">("video");
  const [notes, setNotes] = useState("");
  const [ending, setEnding] = useState(false);
  const endingRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    api
      .report(consultation.session_id)
      .then((loaded) => {
        if (!cancelled) setReport(loaded);
      })
      .catch(() => {
        if (!cancelled) setReportFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [consultation.session_id]);

  usePolling(
    async () => {
      try {
        const latest = await api.consultation(consultation.consultation_id);
        if (!endingRef.current && latest.status === "ended") onClose("The patient ended the consultation");
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) onClose("The consultation is no longer available");
      }
    },
    POLL_MS,
    true,
  );

  const end = async () => {
    endingRef.current = true;
    setEnding(true);
    try {
      await api.endConsultation(consultation.consultation_id);
    } catch {
      toast("The server could not be told the call ended.", "error");
    }
    onClose(null);
  };

  const copyNotes = async () => {
    try {
      await navigator.clipboard.writeText(notes);
      toast("Notes copied.", "success");
    } catch {
      toast("Notes could not be copied.", "error");
    }
  };

  const tabButton = (value: "video" | "report", label: string) => (
    <button
      type="button"
      onClick={() => setTab(value)}
      aria-pressed={tab === value}
      className={`flex-1 rounded-lg py-2 text-sm font-semibold ${tab === value ? "bg-white text-ink-900 shadow-soft dark:bg-ink-700 dark:text-ink-50" : "text-ink-600 dark:text-ink-300"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold text-ink-900 dark:text-ink-50">
        Consultation with {consultation.patient_display_name} · {consultation.exam_title}
      </h1>
      <div className="flex gap-1 rounded-xl bg-ink-100 p-1 lg:hidden dark:bg-ink-800">
        {tabButton("video", "Video")}
        {tabButton("report", "Report")}
      </div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className={`space-y-4 lg:sticky lg:top-20 lg:block lg:self-start ${tab === "video" ? "" : "hidden"}`}>
          <VideoCall channel={consultation.channel_name} remoteName={consultation.patient_display_name} onEnd={() => void end()} ending={ending} />
          {consultation.patient_note && <p className="rounded-xl bg-ink-50 p-3 text-sm text-ink-700 dark:bg-ink-800 dark:text-ink-200">Patient note: “{consultation.patient_note}”</p>}
          <Card
            title="Notes"
            action={
              <Button size="sm" variant="secondary" icon="copy" onClick={() => void copyNotes()} disabled={!notes.trim()}>
                Copy notes
              </Button>
            }
          >
            <label htmlFor="doctor-notes" className="sr-only">
              Consultation notes
            </label>
            <textarea id="doctor-notes" value={notes} onChange={(event) => setNotes(event.target.value)} rows={5} placeholder="Notes stay on this screen and are not saved." className={inputClass} />
          </Card>
        </div>
        <div className={`lg:block ${tab === "report" ? "" : "hidden"}`}>
          {report ? (
            <FullReport report={report} />
          ) : reportFailed ? (
            <p className="rounded-xl bg-red-50 p-4 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">The report could not be loaded.</p>
          ) : (
            <div className="flex justify-center py-10 text-brand-600">
              <Spinner label="Loading report" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Dashboard({ auth, onSignOut, onExpired }: { auth: PortalAuth; onSignOut: () => void; onExpired: () => void }) {
  const [requests, setRequests] = useState<Consultation[]>([]);
  const [active, setActive] = useState<Consultation | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const seen = useRef(new Set<string>());
  const pending = requests.filter((request) => request.status === "requested");
  const ringtone = useRingtone(pending.length > 0 && active === null);

  usePolling(
    async () => {
      try {
        const latest = await api.professionalConsultations(auth.professionalId, auth.code);
        setOffline(false);
        setRequests(latest);
        latest
          .filter((request) => request.status === "requested" && !seen.current.has(request.consultation_id))
          .forEach((request) => {
            seen.current.add(request.consultation_id);
            if ("Notification" in window && Notification.permission === "granted") {
              new Notification("New consultation request", { body: `${request.patient_display_name} · ${request.exam_title}` });
            }
          });
      } catch (error) {
        if (error instanceof ApiError && error.status === 403) onExpired();
        else setOffline(true);
      }
    },
    POLL_MS,
    active === null,
  );

  const respond = async (request: Consultation, accept: boolean) => {
    setBusyId(request.consultation_id);
    try {
      if (accept) {
        setActive(await api.acceptConsultation(request.consultation_id, auth.code));
        setNotice(null);
      } else {
        await api.declineConsultation(request.consultation_id, auth.code);
      }
      setRequests((current) => current.filter((item) => item.consultation_id !== request.consultation_id));
    } catch (error) {
      toast(error instanceof ApiError && error.detail ? error.detail : "The request could not be updated.", "error");
    } finally {
      setBusyId(null);
    }
  };

  const doctorName = auth.doctor?.name ?? "Doctor";

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center gap-4 rounded-2xl border border-ink-100 bg-white p-4 shadow-soft dark:border-ink-800 dark:bg-ink-900">
        {auth.doctor && <DoctorPhoto doctor={auth.doctor} className="w-16" />}
        <div className="flex-1">
          <p className="font-semibold text-ink-900 dark:text-ink-50">{doctorName}</p>
          <p className="flex items-center gap-1.5 text-sm font-medium text-emerald-700 dark:text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Online
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {ringtone.enabled ? (
            <span className="flex items-center gap-1.5 text-sm text-ink-500 dark:text-ink-400">
              <Icon name="speaker" size={18} /> Sound on
            </span>
          ) : (
            <Button size="sm" icon="speaker" onClick={() => void ringtone.enable()}>
              Enable sound
            </Button>
          )}
          {!active && (
            <Button size="sm" variant="ghost" onClick={onSignOut}>
              Sign out
            </Button>
          )}
        </div>
      </header>

      {active ? (
        <CallView
          consultation={active}
          onClose={(message) => {
            setActive(null);
            setNotice(message);
          }}
        />
      ) : (
        <>
          {!ringtone.enabled && (
            <p className="rounded-xl bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-200">Click “Enable sound” once so the ringtone can play when a patient calls.</p>
          )}
          {notice && (
            <p role="status" className="flex items-center justify-between gap-3 rounded-xl bg-ink-100 p-3 text-sm text-ink-800 dark:bg-ink-800 dark:text-ink-100">
              {notice}
              <button type="button" aria-label="Dismiss" onClick={() => setNotice(null)}>
                <Icon name="x" size={16} />
              </button>
            </p>
          )}
          {offline && <p className="rounded-xl bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">Cannot reach the server. Retrying…</p>}
          {pending.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-16 text-center text-ink-500 dark:text-ink-400">
              <Icon name="bell" size={32} />
              <p>Waiting for consultation requests…</p>
            </div>
          ) : (
            <div className="space-y-4">
              {pending.map((request) => (
                <RequestCard
                  key={request.consultation_id}
                  request={request}
                  busy={busyId === request.consultation_id}
                  onAccept={() => void respond(request, true)}
                  onDecline={() => void respond(request, false)}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function PortalPage() {
  const [auth, setAuth] = useState<PortalAuth | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  if (!auth) {
    return (
      <LoginForm
        notice={notice}
        onLogin={(next) => {
          setNotice(null);
          setAuth(next);
        }}
      />
    );
  }
  return (
    <Dashboard
      auth={auth}
      onSignOut={() => setAuth(null)}
      onExpired={() => {
        setNotice("Your access code is no longer valid. Please sign in again.");
        setAuth(null);
      }}
    />
  );
}
