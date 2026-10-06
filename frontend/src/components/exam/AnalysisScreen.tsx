import { motion, useReducedMotion } from "framer-motion";
import { Icon, type IconName } from "../ui/Icon";

const REGION_ICONS: Record<string, IconName[]> = {
  face: ["face", "eye", "mouth"],
  eyes: ["eye", "face", "eye"],
  mouth: ["mouth", "face", "mouth"],
  hands: ["hand", "hands", "hand"],
  movement: ["movement", "hand", "body"],
  posture: ["posture", "body", "movement"],
  skin: ["skin", "face", "hand"],
};

export function AnalysisScreen({ examId, examTitle, stage }: { examId: string; examTitle: string; stage: string | null }) {
  const reduceMotion = useReducedMotion();
  const icons = REGION_ICONS[examId] ?? ["face", "hand", "body"];
  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center gap-8 px-6 text-center" role="status" aria-live="polite">
      <div className="relative h-56 w-56">
        {[0, 1, 2].map((ring) => (
          <motion.div
            key={ring}
            className="absolute inset-0 rounded-full border-2 border-brand-400/50"
            animate={reduceMotion ? undefined : { scale: [0.6, 1.15], opacity: [0.8, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, delay: ring * 0.8, ease: "easeOut" }}
          />
        ))}
        <motion.div
          className="absolute inset-6"
          animate={reduceMotion ? undefined : { rotate: 360 }}
          transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
        >
          {icons.map((icon, index) => {
            const angle = (index / icons.length) * Math.PI * 2;
            return (
              <span
                key={`${icon}-${index}`}
                className="absolute flex h-11 w-11 items-center justify-center rounded-full bg-white text-brand-600 shadow-soft dark:bg-ink-800 dark:text-brand-300"
                style={{ left: `calc(50% + ${Math.cos(angle) * 78}px - 22px)`, top: `calc(50% + ${Math.sin(angle) * 78}px - 22px)` }}
              >
                <Icon name={icon} />
              </span>
            );
          })}
        </motion.div>
        <motion.div
          className="absolute inset-[38%] rounded-full bg-brand-500"
          animate={reduceMotion ? undefined : { scale: [1, 1.18, 1] }}
          transition={{ duration: 1.6, repeat: Infinity }}
        />
      </div>
      <div>
        <h1 className="text-2xl font-semibold text-ink-900 dark:text-ink-50">Analysing results</h1>
        <p className="mt-1 text-sm text-ink-500 dark:text-ink-400">{examTitle}</p>
        <motion.p key={stage ?? "waiting"} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="mt-4 text-base text-brand-700 dark:text-brand-300">
          {stage ?? "Preparing the analysis…"}
        </motion.p>
        <p className="mt-2 text-xs text-ink-400">This usually takes 10 to 60 seconds.</p>
      </div>
    </div>
  );
}
