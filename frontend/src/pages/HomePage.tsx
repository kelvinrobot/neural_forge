import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ExamSummary, HealthResponse, ModelStatusDetail } from "../api/types";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Icon, type IconName } from "../components/ui/Icon";
import { Spinner } from "../components/ui/Spinner";
import { useHealth } from "../hooks/useHealth";
import { titleCase } from "../utils/format";

const EXAM_ICONS: Record<string, IconName> = {
  face: "face",
  eyes: "eye",
  mouth: "mouth",
  hands: "hand",
  movement: "movement",
  posture: "posture",
  skin: "skin",
};

function modelStatusText(value: string | ModelStatusDetail): string {
  if (typeof value === "string") return value;
  if (value.status) return value.status;
  if (value.loaded !== undefined) return value.loaded ? "loaded" : "not loaded";
  return value.error ? "error" : "configured";
}

function gpuName(gpu: HealthResponse["gpu"]): string | null {
  if (!gpu) return null;
  if (typeof gpu === "string") return gpu;
  if (typeof gpu === "boolean") return gpu ? "GPU available" : null;
  return gpu.name ?? (gpu.available ? "GPU available" : null);
}

function errorList(errors: HealthResponse["errors"]): string[] {
  if (!errors) return [];
  return Array.isArray(errors) ? errors : Object.entries(errors).map(([key, value]) => `${titleCase(key)}: ${value}`);
}

function StartingScreen({ unreachable, attempts }: { unreachable: boolean; attempts: number }) {
  const reduceMotion = useReducedMotion();
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 text-center" role="status" aria-live="polite">
      <div className="relative h-28 w-28">
        <motion.div
          className="absolute inset-0 rounded-full border-4 border-brand-200 border-t-brand-600 dark:border-ink-700 dark:border-t-brand-400"
          animate={reduceMotion ? undefined : { rotate: 360 }}
          transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute inset-8 rounded-full bg-brand-500"
          animate={reduceMotion ? undefined : { scale: [1, 1.2, 1], opacity: [0.8, 1, 0.8] }}
          transition={{ duration: 1.8, repeat: Infinity }}
        />
      </div>
      <div>
        <h1 className="text-2xl font-semibold">Starting diagnostic engine</h1>
        <p className="mt-2 max-w-md text-sm text-ink-500 dark:text-ink-400">
          {unreachable
            ? "The server is waking up. This can take a few minutes on the first visit."
            : "Models are loading. This page updates automatically every few seconds."}
        </p>
        <p className="mt-3 text-xs text-ink-400">Checked {attempts} {attempts === 1 ? "time" : "times"}</p>
      </div>
    </div>
  );
}

function ExamCard({ exam }: { exam: ExamSummary }) {
  return (
    <Link
      to={`/exam/${encodeURIComponent(exam.exam_id)}`}
      className="group flex flex-col rounded-2xl border border-ink-100 bg-white p-5 shadow-soft transition-all hover:-translate-y-0.5 hover:border-brand-300 hover:shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500 dark:border-ink-800 dark:bg-ink-900 dark:hover:border-brand-700"
    >
      <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300">
        <Icon name={EXAM_ICONS[exam.exam_id] ?? "camera"} size={26} />
      </span>
      <h2 className="mt-4 text-lg font-semibold text-ink-900 dark:text-ink-50">{exam.title}</h2>
      <p className="mt-1 flex-1 text-sm text-ink-600 dark:text-ink-300">{exam.description}</p>
      <div className="mt-4 flex items-center gap-3 text-xs font-medium text-ink-500 dark:text-ink-400">
        <span>{exam.total_steps} steps</span>
        <span aria-hidden="true">·</span>
        <span>About {exam.estimated_minutes} min</span>
        <span className="ml-auto text-brand-600 transition-transform group-hover:translate-x-1 dark:text-brand-300">
          <Icon name="chevronRight" />
        </span>
      </div>
    </Link>
  );
}

export function HomePage() {
  const { health, unreachable, attempts } = useHealth();
  const [exams, setExams] = useState<ExamSummary[] | null>(null);
  const [examError, setExamError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const engineUp = health !== null && health.status !== "loading" && !unreachable;

  useEffect(() => {
    if (!engineUp || exams) return undefined;
    let cancelled = false;
    setExamError(false);
    api
      .exams()
      .then((response) => {
        if (!cancelled) setExams(response.exams);
      })
      .catch(() => {
        if (!cancelled) setExamError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [engineUp, exams, retryCount]);

  if (!engineUp) return <StartingScreen unreachable={unreachable || health === null} attempts={attempts} />;

  const gpu = gpuName(health.gpu);
  const errors = errorList(health.errors);
  const models = Object.entries(health.models ?? {});

  return (
    <div className="space-y-10">
      <section className="grid gap-6 lg:grid-cols-[1.4fr_1fr] lg:items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-ink-900 dark:text-ink-50 sm:text-4xl">Neural Forge</h1>
          <p className="mt-1 text-lg font-medium text-brand-700 dark:text-brand-300">AI Visual Diagnostics</p>
          <p className="mt-4 max-w-xl text-ink-600 dark:text-ink-300">
            Camera-based examinations guided step by step. Choose an examination to review the steps, set up the camera and begin.
          </p>
        </div>
        <Card
          title="System status"
          action={
            <Badge className={health.status === "ready" ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200" : "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200"}>
              {health.status === "ready" ? "Ready" : "Degraded"}
            </Badge>
          }
        >
          {gpu && (
            <p className="mb-3 flex items-center gap-2 text-sm text-ink-700 dark:text-ink-200">
              <Icon name="movement" size={18} /> {gpu}
            </p>
          )}
          {models.length > 0 && (
            <ul className="grid gap-1.5 text-sm sm:grid-cols-2">
              {models.map(([name, value]) => (
                <li key={name} className="flex items-center justify-between gap-2 rounded-lg bg-ink-50 px-2.5 py-1.5 dark:bg-ink-800">
                  <span className="truncate text-ink-700 dark:text-ink-200">{titleCase(name)}</span>
                  <span className="text-xs font-medium text-ink-500 dark:text-ink-400">{modelStatusText(value)}</span>
                </li>
              ))}
            </ul>
          )}
          {errors.length > 0 && (
            <ul className="mt-3 space-y-1 text-xs text-amber-800 dark:text-amber-200">
              {errors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      <section aria-labelledby="exams-heading">
        <h2 id="exams-heading" className="mb-4 text-xl font-semibold text-ink-900 dark:text-ink-50">
          Examinations
        </h2>
        {examError && (
          <div className="flex items-center gap-3 rounded-2xl bg-red-50 p-4 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            The examination list could not be loaded.
            <Button size="sm" variant="secondary" onClick={() => setRetryCount((count) => count + 1)}>
              Retry
            </Button>
          </div>
        )}
        {!exams && !examError && (
          <div className="flex justify-center py-10 text-brand-600">
            <Spinner size={32} label="Loading examinations" />
          </div>
        )}
        {exams && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {exams.map((exam) => (
              <ExamCard key={exam.exam_id} exam={exam} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
