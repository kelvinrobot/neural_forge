import { useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { DoctorPhoto } from "../components/consult/DoctorPhoto";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Icon } from "../components/ui/Icon";
import { inputClass, labelClass } from "../components/ui/fieldStyles";
import { findDoctor } from "../data/doctors";

const SHARED_ITEMS = ["Findings from your examination", "Measurements", "Evidence images", "Your age and sex"];

export function ConsultRequestPage() {
  const { sessionId = "", doctorId = "" } = useParams();
  const navigate = useNavigate();
  const doctor = findDoctor(doctorId);
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [consent, setConsent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const directory = `/consult/${encodeURIComponent(sessionId)}`;

  if (!doctor?.live) return <Navigate to={directory} replace />;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!consent || submitting) return;
    setSubmitting(true);
    setFailure(null);
    try {
      const consultation = await api.createConsultation({
        session_id: sessionId,
        professional_id: doctor.id,
        professional_name: doctor.name,
        patient_display_name: name.trim() || "Patient",
        patient_note: note.trim(),
        consent: true,
      });
      navigate(`/consultation/${encodeURIComponent(consultation.consultation_id)}`);
    } catch (error) {
      setFailure(error instanceof ApiError && error.detail ? error.detail : "The request could not be sent. Please try again.");
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <Link to={directory} className="text-sm font-medium text-brand-700 hover:underline dark:text-brand-300">
        ← All doctors
      </Link>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
        <DoctorPhoto doctor={doctor} className="w-36 sm:w-44" />
        <div>
          <h1 className="text-2xl font-bold text-ink-900 dark:text-ink-50">{doctor.name}</h1>
          <p className="font-medium text-brand-700 dark:text-brand-300">{doctor.specialty}</p>
          <p className="mt-2 text-sm text-ink-600 dark:text-ink-300">{doctor.bio}</p>
        </div>
      </div>
      <form onSubmit={submit} className="space-y-5">
        <Card>
          <div className="space-y-4">
            <div>
              <label htmlFor="patient-name" className={labelClass}>
                Your name <span className="font-normal text-ink-500">(optional)</span>
              </label>
              <input id="patient-name" value={name} onChange={(event) => setName(event.target.value)} maxLength={80} placeholder="Patient" className={inputClass} />
            </div>
            <div>
              <label htmlFor="patient-note" className={labelClass}>
                Short note for the doctor <span className="font-normal text-ink-500">(optional)</span>
              </label>
              <textarea
                id="patient-note"
                value={note}
                onChange={(event) => setNote(event.target.value)}
                maxLength={500}
                rows={3}
                placeholder="For example, what you would like to discuss"
                className={inputClass}
              />
            </div>
          </div>
        </Card>
        <Card title="What will be shared with this doctor">
          <ul className="space-y-1.5 text-sm text-ink-700 dark:text-ink-200">
            {SHARED_ITEMS.map((item) => (
              <li key={item} className="flex items-center gap-2">
                <Icon name="check" size={16} className="text-brand-600" />
                {item}
              </li>
            ))}
          </ul>
          <label className="mt-4 flex items-start gap-3 rounded-xl bg-ink-50 p-3 text-sm font-medium text-ink-800 dark:bg-ink-800 dark:text-ink-100">
            <input type="checkbox" required checked={consent} onChange={(event) => setConsent(event.target.checked)} className="mt-0.5 h-4 w-4 accent-brand-600" />
            I agree to share this report with this doctor.
          </label>
        </Card>
        {failure && (
          <p role="alert" className="rounded-xl bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            {failure}
          </p>
        )}
        <Button type="submit" size="lg" icon="video" className="w-full sm:w-auto" disabled={!consent || submitting}>
          {submitting ? "Sending request…" : "Request consultation"}
        </Button>
      </form>
    </div>
  );
}
