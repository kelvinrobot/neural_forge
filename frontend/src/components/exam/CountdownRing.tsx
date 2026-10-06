import { AnimatePresence, motion } from "framer-motion";

const SIZE = 168;
const STROKE = 10;
const RADIUS = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function CountdownRing({ value }: { value: number | null }) {
  return (
    <AnimatePresence>
      {value !== null && value > 0 && (
        <motion.div
          className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/25"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          aria-live="assertive"
        >
          <div className="relative" style={{ width: SIZE, height: SIZE }}>
            <svg width={SIZE} height={SIZE} className="-rotate-90">
              <circle cx={SIZE / 2} cy={SIZE / 2} r={RADIUS} fill="rgba(0,0,0,0.45)" stroke="rgba(255,255,255,0.25)" strokeWidth={STROKE} />
              <motion.circle
                key={value}
                cx={SIZE / 2}
                cy={SIZE / 2}
                r={RADIUS}
                fill="none"
                stroke="#5eead4"
                strokeWidth={STROKE}
                strokeLinecap="round"
                strokeDasharray={CIRCUMFERENCE}
                initial={{ strokeDashoffset: 0 }}
                animate={{ strokeDashoffset: CIRCUMFERENCE }}
                transition={{ duration: 1, ease: "linear" }}
              />
            </svg>
            <motion.span
              key={`n-${value}`}
              className="absolute inset-0 flex items-center justify-center text-7xl font-bold text-white"
              initial={{ scale: 1.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              {value}
            </motion.span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
