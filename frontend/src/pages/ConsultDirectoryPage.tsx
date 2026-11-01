import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { DoctorPhoto } from "../components/consult/DoctorPhoto";
import { Button } from "../components/ui/Button";
import { Icon } from "../components/ui/Icon";
import { inputClass } from "../components/ui/fieldStyles";
import { doctors, type Doctor } from "../data/doctors";

const ordered = [...doctors].sort((a, b) => Number(b.live) - Number(a.live));
const specialties = Array.from(new Set(ordered.map((doctor) => doctor.specialty)));

function DoctorCard({ doctor, sessionId }: { doctor: Doctor; sessionId: string }) {
  return (
    <article className="flex flex-col rounded-2xl border border-ink-100 bg-white p-4 shadow-soft dark:border-ink-800 dark:bg-ink-900">
      <DoctorPhoto doctor={doctor} />
      <div className="mt-3 flex items-start justify-between gap-2">
        <div>
          <h2 className="font-semibold text-ink-900 dark:text-ink-50">{doctor.name}</h2>
          <p className="text-sm font-medium text-brand-700 dark:text-brand-300">{doctor.specialty}</p>
        </div>
        <span className={`flex shrink-0 items-center gap-1.5 text-xs font-semibold ${doctor.live ? "text-emerald-700 dark:text-emerald-300" : "text-ink-400"}`}>
          <span className={`h-2 w-2 rounded-full ${doctor.live ? "bg-emerald-500" : "bg-ink-300 dark:bg-ink-600"}`} />
          {doctor.live ? "Available now" : "Offline"}
        </span>
      </div>
      <p className="mt-2 flex-1 text-sm text-ink-600 dark:text-ink-300">{doctor.bio}</p>
      {doctor.live ? (
        <Link
          to={`/consult/${encodeURIComponent(sessionId)}/request/${doctor.id}`}
          className="mt-4 inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-brand-600 px-4 text-sm font-semibold text-white shadow-soft hover:bg-brand-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500"
        >
          <Icon name="video" size={18} />
          Request consultation
        </Link>
      ) : (
        <Button variant="secondary" className="mt-4" disabled>
          Offline
        </Button>
      )}
    </article>
  );
}

export function ConsultDirectoryPage() {
  const { sessionId = "" } = useParams();
  const [query, setQuery] = useState("");
  const [specialty, setSpecialty] = useState<string | null>(null);

  const visible = useMemo(() => {
    const text = query.trim().toLowerCase();
    return ordered.filter(
      (doctor) =>
        (!specialty || doctor.specialty === specialty) &&
        (!text || doctor.name.toLowerCase().includes(text) || doctor.specialty.toLowerCase().includes(text)),
    );
  }, [query, specialty]);

  const chip = (label: string, value: string | null) => (
    <button
      key={label}
      type="button"
      onClick={() => setSpecialty(value)}
      aria-pressed={specialty === value}
      className={`rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
        specialty === value
          ? "bg-brand-600 text-white"
          : "bg-white text-ink-700 ring-1 ring-ink-200 hover:bg-ink-50 dark:bg-ink-900 dark:text-ink-200 dark:ring-ink-700"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="space-y-6">
      <div>
        <Link to={`/report/${encodeURIComponent(sessionId)}`} className="text-sm font-medium text-brand-700 hover:underline dark:text-brand-300">
          ← Back to report
        </Link>
        <h1 className="mt-2 text-2xl font-bold text-ink-900 dark:text-ink-50 sm:text-3xl">Consult a doctor</h1>
        <p className="mt-1 text-ink-600 dark:text-ink-300">Choose an available doctor to review your report with you on a video call.</p>
      </div>
      <div className="space-y-3">
        <div className="relative max-w-md">
          <Icon name="search" size={18} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
          <label htmlFor="doctor-search" className="sr-only">
            Search doctors
          </label>
          <input
            id="doctor-search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by name or specialty"
            className={`${inputClass} pl-9`}
          />
        </div>
        <div className="flex flex-wrap gap-2">
          {chip("All", null)}
          {specialties.map((name) => chip(name, name))}
        </div>
      </div>
      {visible.length === 0 ? (
        <p className="py-10 text-center text-sm text-ink-500">No doctors match your search.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {visible.map((doctor) => (
            <DoctorCard key={doctor.id} doctor={doctor} sessionId={sessionId} />
          ))}
        </div>
      )}
    </div>
  );
}
