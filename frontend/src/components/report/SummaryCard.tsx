import type { Interpretation } from "../../api/types";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";

interface SummaryCardProps {
  interpretation: Interpretation;
  onReadAloud: () => void;
  readAloudAvailable: boolean;
}

export function SummaryCard({ interpretation, onReadAloud, readAloudAvailable }: SummaryCardProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="Clinician summary">
        <p className="whitespace-pre-line text-sm leading-relaxed text-ink-700 dark:text-ink-200">{interpretation.summary}</p>
      </Card>
      <Card
        title="Message for the patient"
        action={
          readAloudAvailable && (
            <Button variant="secondary" size="sm" icon="speaker" onClick={onReadAloud} className="print:hidden">
              Read aloud
            </Button>
          )
        }
      >
        <p className="whitespace-pre-line text-sm leading-relaxed text-ink-700 dark:text-ink-200">{interpretation.patient_message}</p>
      </Card>
    </div>
  );
}
