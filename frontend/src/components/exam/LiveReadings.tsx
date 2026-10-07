import { AnimatePresence, motion } from "framer-motion";
import { isBilateral, type LiveValue } from "../../api/types";
import { formatValue } from "../../utils/format";

function describe(live: LiveValue): string {
  if (isBilateral(live)) {
    return `${live.label} R ${formatValue(live.right, live.unit)} / L ${formatValue(live.left, live.unit)}`;
  }
  if (typeof live.value === "string") {
    return `${live.label}: ${live.value.charAt(0).toUpperCase()}${live.value.slice(1)}`;
  }
  return `${live.label} ${formatValue(live.value, live.unit)}`;
}

export function LiveReadings({ live }: { live: LiveValue | null }) {
  return (
    <div aria-live="polite" className="min-h-[28px]">
      <AnimatePresence mode="popLayout">
        {live && (
          <motion.span
            key={live.label}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="inline-flex items-center gap-1.5 rounded-full bg-black/55 px-3 py-1 text-xs font-medium text-white backdrop-blur"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-brand-300" />
            {describe(live)}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
}
