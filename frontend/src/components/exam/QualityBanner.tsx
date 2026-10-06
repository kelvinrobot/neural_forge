import { AnimatePresence, motion } from "framer-motion";
import type { QualityCode } from "../../api/types";
import { qualityHintFor } from "../../utils/qualityHints";
import { Icon } from "../ui/Icon";

export function QualityBanner({ code }: { code: QualityCode | null }) {
  const hint = code ? qualityHintFor(code) : null;
  return (
    <div className="pointer-events-none absolute inset-x-0 top-14 flex justify-center px-3" aria-live="polite">
      <AnimatePresence>
        {hint && (
          <motion.div
            key={code}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex max-w-md items-center gap-3 rounded-2xl bg-amber-400/95 px-4 py-2.5 text-amber-950 shadow-lg"
            role="status"
          >
            <motion.span animate={{ scale: [1, 1.12, 1] }} transition={{ duration: 1.4, repeat: Infinity }}>
              <Icon name={hint.icon} size={26} />
            </motion.span>
            <span>
              <span className="block text-sm font-semibold">{hint.title}</span>
              <span className="block text-xs">{hint.hint}</span>
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
