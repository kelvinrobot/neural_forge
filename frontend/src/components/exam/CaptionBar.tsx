import { AnimatePresence, motion } from "framer-motion";

export function CaptionBar({ text }: { text: string }) {
  return (
    <div className="min-h-[52px] rounded-xl bg-ink-900 px-4 py-3 text-sm text-white dark:bg-black" aria-live="polite" aria-label="Spoken guidance captions">
      <AnimatePresence mode="wait">
        <motion.p key={text} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
          {text || "Spoken guidance will appear here."}
        </motion.p>
      </AnimatePresence>
    </div>
  );
}
