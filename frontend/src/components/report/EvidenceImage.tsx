import { useState } from "react";
import { evidenceSource } from "../../utils/format";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";

export function EvidenceImage({ image, title }: { image: string; title: string }) {
  const [open, setOpen] = useState(false);
  const source = evidenceSource(image);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group block overflow-hidden rounded-xl border border-ink-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500 dark:border-ink-800"
        aria-label={`Enlarge evidence image for ${title}`}
      >
        <img src={source} alt={`Evidence for ${title}`} className="max-h-48 w-full object-cover transition-transform group-hover:scale-105 print:max-h-60" />
      </button>
      <Modal open={open} title={title} onClose={() => setOpen(false)} wide footer={<Button onClick={() => setOpen(false)}>Close</Button>}>
        <img src={source} alt={`Evidence for ${title}, enlarged`} className="w-full rounded-xl" />
      </Modal>
    </>
  );
}
