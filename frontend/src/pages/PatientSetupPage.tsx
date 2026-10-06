import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { ExamDetail, PatientInfo, Sex } from "../api/types";
import { FramingGuide } from "../components/exam/FramingGuide";
import { SKIN_TONES, type SkinTone } from "../components/guide/motion";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Icon } from "../components/ui/Icon";
import { Spinner } from "../components/ui/Spinner";
import { toast } from "../components/ui/Toast";
import { useCamera } from "../hooks/useCamera";
import { pickDefaultVoice, useSpeech, useVoices } from "../hooks/useSpeech";
import { useExamStore } from "../state/examStore";
import { useSettingsStore } from "../state/settingsStore";

const inputClass =
  "h-11 w-full rounded-xl border border-ink-200 bg-white px-3 text-sm text-ink-900 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:border-ink-700 dark:bg-ink-800 dark:text-ink-50 dark:focus:ring-brand-900";

function TagInput({ id, label, values, onChange, placeholder }: { id: string; label: string; values: string[]; onChange: (values: string[]) => void; placeholder: string }) {
  const [draft, setDraft] = useState("");
  const commit = () => {
    const value = draft.trim().replace(/,$/, "");
    if (value && !values.includes(value)) onChange([...values, value]);
    setDraft("");
  };
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      commit();
    } else if (event.key === "Backspace" && !draft && values.length > 0) {
      onChange(values.slice(0, -1));
    }
  };
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm font-medium text-ink-700 dark:text-ink-200">
        {label}
      </label>
      <div className="flex min-h-11 flex-wrap items-center gap-1.5 rounded-xl border border-ink-200 bg-white px-2 py-1.5 focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-200 dark:border-ink-700 dark:bg-ink-800">
        {values.map((value) => (
          <span key={value} className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-800 dark:bg-brand-950 dark:text-brand-200">
            {value}
            <button type="button" aria-label={`Remove ${value}`} onClick={() => onChange(values.filter((item) => item !== value))} className="rounded-full hover:text-brand-950">
              <Icon name="x" size={12} />
            </button>
          </span>
        ))}
        <input
          id={id}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
          onBlur={commit}
          placeholder={values.length === 0 ? placeholder : "Add more"}
          className="min-w-[8rem] flex-1 bg-transparent px-1 text-sm text-ink-900 outline-none dark:text-ink-50"
        />
      </div>
      <p className="mt-1 text-xs text-ink-400">Press Enter to add each item.</p>
    </div>
  );
}

type Lighting = "checking" | "good" | "dark" | "bright";

function useLightingCheck(video: HTMLVideoElement | null, active: boolean): Lighting {
  const [lighting, setLighting] = useState<Lighting>("checking");
  useEffect(() => {
    if (!video || !active) return undefined;
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 36;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return undefined;
    const timer = window.setInterval(() => {
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
      let total = 0;
      for (let index = 0; index < data.length; index += 4) {
        total += 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
      }
      const mean = total / (data.length / 4);
      setLighting(mean < 60 ? "dark" : mean > 205 ? "bright" : "good");
    }, 1000);
    return () => window.clearInterval(timer);
  }, [video, active]);
  return lighting;
}

const LIGHTING_TEXT: Record<Lighting, { text: string; className: string }> = {
  checking: { text: "Checking the lighting…", className: "text-ink-500" },
  good: { text: "Lighting looks good.", className: "text-emerald-700 dark:text-emerald-300" },
  dark: { text: "The room looks dark. Turn on a light or face a window.", className: "text-amber-700 dark:text-amber-300" },
  bright: { text: "The image looks very bright. Avoid direct sunlight on the camera.", className: "text-amber-700 dark:text-amber-300" },
};

export function PatientSetupPage() {
  const { examId = "" } = useParams();
  const navigate = useNavigate();
  const [exam, setExam] = useState<ExamDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [age, setAge] = useState("");
  const [sex, setSex] = useState<Sex>("unspecified");
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [notes, setNotes] = useState("");
  const [consent, setConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const cameraDeviceId = useSettingsStore((state) => state.cameraDeviceId);
  const setCameraDeviceId = useSettingsStore((state) => state.setCameraDeviceId);
  const voiceEnabled = useSettingsStore((state) => state.voiceEnabled);
  const setVoiceEnabled = useSettingsStore((state) => state.setVoiceEnabled);
  const voiceURI = useSettingsStore((state) => state.voiceURI);
  const setVoiceURI = useSettingsStore((state) => state.setVoiceURI);
  const skinTone = useSettingsStore((state) => state.guideSkinTone);
  const setSkinTone = useSettingsStore((state) => state.setGuideSkinTone);
  const setStoreExam = useExamStore((state) => state.setExam);
  const setPatient = useExamStore((state) => state.setPatient);
  const beginSession = useExamStore((state) => state.beginSession);

  const camera = useCamera(cameraDeviceId);
  const voices = useVoices();
  const { speak } = useSpeech();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const lighting = useLightingCheck(videoElement, camera.status === "ready");

  useEffect(() => {
    let cancelled = false;
    api
      .exam(examId)
      .then((detail) => {
        if (cancelled) return;
        setExam(detail);
        setStoreExam(detail);
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoadError(error instanceof ApiError && error.status === 404 ? "This examination does not exist." : "The examination could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, [examId, setStoreExam]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = camera.stream;
    if (camera.stream) void video.play().catch(() => undefined);
    setVideoElement(video);
  }, [camera.stream]);

  const englishDefault = pickDefaultVoice(voices);
  const selectedVoice = voiceURI ?? englishDefault?.voiceURI ?? "";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!exam || !consent) return;
    const parsedAge = age.trim() === "" ? null : Number.parseInt(age, 10);
    const patient: PatientInfo = {
      age: parsedAge !== null && Number.isFinite(parsedAge) ? parsedAge : null,
      sex,
      symptoms,
      medical_history: history,
      notes: notes.trim(),
    };
    setSubmitting(true);
    try {
      const session = await api.createSession({ exam_id: exam.exam_id, consent: true, camera_mirrored: false, patient });
      setPatient(patient);
      beginSession(session);
      camera.stop();
      navigate(`/exam/${encodeURIComponent(exam.exam_id)}/live`);
    } catch (error) {
      toast(error instanceof ApiError ? `The examination could not be started: ${error.detail}` : "The examination could not be started.", "error");
    } finally {
      setSubmitting(false);
    }
  };

  if (loadError) {
    return (
      <div className="flex flex-col items-center gap-4 py-20 text-center">
        <p className="text-lg">{loadError}</p>
        <Button onClick={() => navigate("/")}>Back to examinations</Button>
      </div>
    );
  }
  if (!exam) {
    return (
      <div className="flex justify-center py-20 text-brand-600">
        <Spinner size={36} label="Loading examination" />
      </div>
    );
  }

  const firstView = exam.steps[0]?.view ?? "face";

  return (
    <form onSubmit={submit} className="space-y-6">
      <div>
        <button type="button" onClick={() => navigate("/")} className="text-sm font-medium text-brand-700 hover:underline dark:text-brand-300">
          ← All examinations
        </button>
        <h1 className="mt-2 text-3xl font-bold text-ink-900 dark:text-ink-50">{exam.title}</h1>
        <p className="mt-1 max-w-2xl text-ink-600 dark:text-ink-300">{exam.description}</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
        <div className="space-y-6">
          <Card title="Patient details">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="age" className="mb-1.5 block text-sm font-medium text-ink-700 dark:text-ink-200">
                  Age
                </label>
                <input id="age" type="number" min={0} max={120} inputMode="numeric" value={age} onChange={(event) => setAge(event.target.value)} className={inputClass} />
              </div>
              <div>
                <label htmlFor="sex" className="mb-1.5 block text-sm font-medium text-ink-700 dark:text-ink-200">
                  Sex
                </label>
                <select id="sex" value={sex} onChange={(event) => setSex(event.target.value as Sex)} className={inputClass}>
                  <option value="unspecified">Prefer not to say</option>
                  <option value="female">Female</option>
                  <option value="male">Male</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div className="sm:col-span-2">
                <TagInput id="symptoms" label="Symptoms" values={symptoms} onChange={setSymptoms} placeholder="For example: facial droop" />
              </div>
              <div className="sm:col-span-2">
                <TagInput id="history" label="Medical history" values={history} onChange={setHistory} placeholder="For example: diabetes" />
              </div>
              <div className="sm:col-span-2">
                <label htmlFor="notes" className="mb-1.5 block text-sm font-medium text-ink-700 dark:text-ink-200">
                  Notes
                </label>
                <textarea
                  id="notes"
                  rows={3}
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  className={`${inputClass} h-auto py-2`}
                />
              </div>
            </div>
          </Card>

          <Card title={`What happens (${exam.total_steps} steps, about ${exam.estimated_minutes} min)`}>
            <ol className="space-y-2">
              {exam.steps.map((step) => (
                <li key={step.id} className="flex gap-3 text-sm">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-50 text-xs font-semibold text-brand-700 dark:bg-brand-950 dark:text-brand-300">
                    {step.number}
                  </span>
                  <span>
                    <span className="font-medium text-ink-900 dark:text-ink-50">{step.title}</span>
                    {step.skippable && <span className="ml-2 text-xs text-ink-400">optional</span>}
                  </span>
                </li>
              ))}
            </ol>
          </Card>
        </div>

        <div className="space-y-6">
          <Card title="Camera setup">
            <div className="relative aspect-video overflow-hidden rounded-xl bg-ink-950">
              <video ref={videoRef} className="h-full w-full -scale-x-100 object-cover" playsInline muted autoPlay aria-label="Camera preview" />
              {camera.status === "ready" && <FramingGuide view={firstView} ok={lighting === "good"} />}
              {camera.status === "requesting" && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <Spinner label="Starting camera" />
                </div>
              )}
              {camera.status !== "ready" && camera.status !== "requesting" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center text-white">
                  <Icon name="camera" size={32} />
                  <p className="text-sm">{camera.message}</p>
                  <Button size="sm" variant="secondary" onClick={camera.retry}>
                    Try again
                  </Button>
                </div>
              )}
            </div>
            {camera.status === "ready" && <p className={`mt-3 text-sm ${LIGHTING_TEXT[lighting].className}`}>{LIGHTING_TEXT[lighting].text}</p>}
            {camera.devices.length > 1 && (
              <div className="mt-3">
                <label htmlFor="camera" className="mb-1.5 block text-sm font-medium text-ink-700 dark:text-ink-200">
                  Camera
                </label>
                <select id="camera" value={cameraDeviceId ?? camera.devices[0]?.deviceId ?? ""} onChange={(event) => setCameraDeviceId(event.target.value)} className={inputClass}>
                  {camera.devices.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </Card>

          <Card title="Voice guidance">
            <label className="flex items-center justify-between gap-3 text-sm text-ink-700 dark:text-ink-200">
              Spoken instructions
              <input type="checkbox" checked={voiceEnabled} onChange={(event) => setVoiceEnabled(event.target.checked)} className="h-5 w-5 accent-brand-600" />
            </label>
            {voices.length > 0 && (
              <div className="mt-3 flex gap-2">
                <label htmlFor="voice" className="sr-only">
                  Voice
                </label>
                <select id="voice" value={selectedVoice} onChange={(event) => setVoiceURI(event.target.value)} className={inputClass} disabled={!voiceEnabled}>
                  {voices.map((voice) => (
                    <option key={voice.voiceURI} value={voice.voiceURI}>
                      {voice.name} ({voice.lang})
                    </option>
                  ))}
                </select>
                <Button variant="secondary" icon="speaker" disabled={!voiceEnabled} onClick={() => speak("This is how the voice guidance will sound.", { interrupt: true })}>
                  Test
                </Button>
              </div>
            )}
            <fieldset className="mt-4">
              <legend className="mb-2 text-sm font-medium text-ink-700 dark:text-ink-200">Guide figure skin tone</legend>
              <div className="flex gap-2">
                {(Object.keys(SKIN_TONES) as SkinTone[]).map((tone) => (
                  <button
                    key={tone}
                    type="button"
                    onClick={() => setSkinTone(tone)}
                    aria-label={`Skin tone ${tone}`}
                    aria-pressed={skinTone === tone}
                    className={`h-9 w-9 rounded-full border-2 ${skinTone === tone ? "border-brand-600 ring-2 ring-brand-200" : "border-white dark:border-ink-700"}`}
                    style={{ backgroundColor: SKIN_TONES[tone] }}
                  />
                ))}
              </div>
            </fieldset>
          </Card>

          <Card>
            <label className="flex items-start gap-3 text-sm text-ink-800 dark:text-ink-100">
              <input type="checkbox" required checked={consent} onChange={(event) => setConsent(event.target.checked)} className="mt-0.5 h-5 w-5 shrink-0 accent-brand-600" />
              I agree to camera-based examination. Video is processed in real time and is not stored.
            </label>
            <Button type="submit" size="lg" icon="play" className="mt-4 w-full" disabled={!consent || submitting || camera.status !== "ready"}>
              {submitting ? "Starting…" : "Begin examination"}
            </Button>
            {camera.status !== "ready" && <p className="mt-2 text-center text-xs text-ink-500">A working camera is needed to begin.</p>}
          </Card>
        </div>
      </div>
    </form>
  );
}
