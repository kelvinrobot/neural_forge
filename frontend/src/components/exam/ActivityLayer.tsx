import { motion, useReducedMotion } from "framer-motion";
import { useExamStore } from "../../state/examStore";
import { LiveReadings } from "./LiveReadings";
import { Sparkline } from "./Sparkline";

export function ActivityLayer({ stepTitle }: { stepTitle: string }) {
  const phase = useExamStore((state) => state.phase);
  const paused = useExamStore((state) => state.paused);
  const progress = useExamStore((state) => state.progress);
  const qualityOk = useExamStore((state) => state.qualityOk);
  const live = useExamStore((state) => state.live);
  const history = useExamStore((state) => state.liveHistory);
  const reduceMotion = useReducedMotion();
  const collecting = phase === "collecting" && !paused;

  return (
    <div className="pointer-events-none absolute inset-0">
      {collecting && !reduceMotion && (
        <motion.div
          className="absolute inset-x-0 h-20 bg-gradient-to-b from-transparent via-brand-300/25 to-transparent"
          initial={{ top: "-15%" }}
          animate={{ top: ["-15%", "100%"] }}
          transition={{ duration: 2.6, repeat: Infinity, ease: "linear" }}
          aria-hidden="true"
        />
      )}
      {collecting && (
        <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs font-semibold text-white backdrop-blur">
          <motion.span
            className="h-2 w-2 rounded-full bg-brand-300"
            animate={reduceMotion ? undefined : { opacity: [1, 0.25, 1], scale: [1, 1.3, 1] }}
            transition={{ duration: 1.2, repeat: Infinity }}
          />
          Measuring · {stepTitle}
        </div>
      )}
      <div className="absolute right-3 top-3 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs font-medium text-white backdrop-blur">
        <span className={`h-2 w-2 rounded-full ${qualityOk ? "bg-emerald-400" : "bg-amber-400"}`} />
        <span>{qualityOk ? "Good view" : "Adjusting"}</span>
        {progress && <span className="tabular-nums text-white/80">{Math.round(progress.fps)} fps</span>}
      </div>
      {collecting && (
        <div className="absolute bottom-3 left-3 flex flex-col items-start gap-2">
          <Sparkline points={history} />
          <LiveReadings live={live} />
        </div>
      )}
    </div>
  );
}
