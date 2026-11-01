import { useState, type FormEvent } from "react";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Icon } from "../components/ui/Icon";
import { inputClass, labelClass } from "../components/ui/fieldStyles";

const TEXT_FIELDS = [
  { id: "fullName", label: "Full name", type: "text" },
  { id: "specialty", label: "Specialty", type: "text" },
  { id: "licence", label: "Licence number", type: "text" },
  { id: "council", label: "Issuing council", type: "text" },
  { id: "country", label: "Country", type: "text" },
  { id: "experience", label: "Years of experience", type: "number" },
  { id: "workplace", label: "Workplace", type: "text" },
  { id: "languages", label: "Languages", type: "text" },
] as const;

const PROFESSIONS = ["Medical doctor", "Psychologist", "Physiotherapist", "Nurse", "Other health professional"];

const STEPS = ["Identity check", "Licence check with the issuing council", "Interview", "Approved"];

const HOW_IT_WORKS = [
  "Register with your professional details and upload your certificate and a photo ID.",
  "We verify your identity and confirm your licence directly with the issuing council.",
  "After a short interview your profile goes live and patients can request consultations.",
];

export function JoinDoctorPage() {
  const [submitted, setSubmitted] = useState(false);
  const [files, setFiles] = useState<{ certificate: File | null; identity: File | null }>({ certificate: null, identity: null });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    setSubmitted(true);
  };

  const fileInput = (key: "certificate" | "identity", label: string) => (
    <div>
      <label htmlFor={key} className={labelClass}>
        {label}
      </label>
      <label
        htmlFor={key}
        className="flex cursor-pointer items-center gap-2 rounded-xl border border-dashed border-ink-300 px-3 py-3 text-sm text-ink-600 hover:border-brand-500 dark:border-ink-600 dark:text-ink-300"
      >
        <Icon name="upload" size={18} />
        <span className="truncate">{files[key]?.name ?? "Choose a file"}</span>
      </label>
      <input
        id={key}
        type="file"
        accept="image/*,application/pdf"
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0] ?? null;
          setFiles((current) => ({ ...current, [key]: file }));
        }}
      />
    </div>
  );

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-ink-900 dark:text-ink-50 sm:text-3xl">Join as a doctor</h1>
        <p className="mt-1 text-ink-600 dark:text-ink-300">Registration opens after launch. Here is how joining will work.</p>
      </div>
      <ol className="grid gap-3 sm:grid-cols-3">
        {HOW_IT_WORKS.map((text, index) => (
          <li key={text} className="rounded-2xl border border-ink-100 bg-white p-4 text-sm text-ink-700 dark:border-ink-800 dark:bg-ink-900 dark:text-ink-200">
            <span className="mb-2 flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-xs font-bold text-white">{index + 1}</span>
            {text}
          </li>
        ))}
      </ol>

      {submitted ? (
        <Card>
          <p className="font-medium text-ink-900 dark:text-ink-50" role="status">
            Thank you. Registration opens after launch. Every doctor's identity and licence will be verified before their profile goes live.
          </p>
          <ol className="mt-6 space-y-4">
            {STEPS.map((step, index) => (
              <li key={step} className="flex items-center gap-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2 border-brand-500 text-sm font-bold text-brand-700 dark:text-brand-300">
                  {index + 1}
                </span>
                <span className="text-sm text-ink-800 dark:text-ink-100">{step}</span>
              </li>
            ))}
          </ol>
        </Card>
      ) : (
        <Card title="Registration preview">
          <form onSubmit={submit} className="grid gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="profession" className={labelClass}>
                Profession
              </label>
              <select id="profession" required defaultValue="" className={inputClass}>
                <option value="" disabled>
                  Select
                </option>
                {PROFESSIONS.map((profession) => (
                  <option key={profession}>{profession}</option>
                ))}
              </select>
            </div>
            {TEXT_FIELDS.map((field) => (
              <div key={field.id}>
                <label htmlFor={field.id} className={labelClass}>
                  {field.label}
                </label>
                <input id={field.id} type={field.type} min={field.type === "number" ? 0 : undefined} required className={inputClass} />
              </div>
            ))}
            {fileInput("certificate", "Certificate")}
            {fileInput("identity", "Photo ID")}
            <Button type="submit" className="sm:col-span-2 sm:justify-self-start">
              Submit registration
            </Button>
          </form>
        </Card>
      )}
    </div>
  );
}
