import { motion } from "framer-motion";
import type { StepView } from "../../api/types";

interface FramingGuideProps {
  view: StepView;
  ok: boolean;
}

const HAND = "c0-6 4-8 7-8s4 3 4 6v-14c0-3 2-5 4-5s4 2 4 5v-4c0-3 2-5 4-5s4 2 4 5v4c0-3 2-4 4-4s4 2 4 5v10c0-3 2-4 4-4s3 2 3 5v14c0 8-6 16-16 16h-4c-8 0-14-6-16-14z";

function Shapes({ view }: { view: StepView }) {
  switch (view) {
    case "face":
      return <ellipse cx="80" cy="45" rx="17" ry="24" />;
    case "face_close":
      return (
        <>
          <rect x="24" y="30" width="112" height="26" rx="13" />
          <ellipse cx="58" cy="43" rx="14" ry="7" />
          <ellipse cx="102" cy="43" rx="14" ry="7" />
        </>
      );
    case "mouth":
      return (
        <>
          <rect x="46" y="28" width="68" height="36" rx="16" />
          <path d="M60 46q20 10 40 0" />
        </>
      );
    case "hands":
      return (
        <>
          <path d={`M30 60${HAND}`} transform="translate(0 -6)" />
          <path d={`M98 60${HAND}`} transform="translate(0 -6)" />
        </>
      );
    case "hand_close":
      return <path d={`M64 64${HAND}`} transform="translate(-28 -50) scale(1.6)" />;
    case "upper_body":
      return (
        <>
          <ellipse cx="80" cy="26" rx="10" ry="13" />
          <path d="M44 88c2-28 16-40 36-42 20 2 34 14 36 42" />
        </>
      );
    case "full_body":
      return (
        <>
          <circle cx="80" cy="12" r="5" />
          <path d="M80 18v30M68 26l12 6 12-6M80 48l-8 36M80 48l8 36" />
        </>
      );
    case "skin":
      return <path d="M30 18h-8v12M130 18h8v12M30 72h-8V60M130 72h8V60" />;
  }
}

export function FramingGuide({ view, ok }: FramingGuideProps) {
  return (
    <svg
      viewBox="0 0 160 90"
      preserveAspectRatio="xMidYMid meet"
      className="pointer-events-none absolute inset-0 h-full w-full"
      aria-hidden="true"
    >
      <motion.g
        fill="none"
        strokeWidth="0.9"
        strokeLinecap="round"
        strokeLinejoin="round"
        animate={{ stroke: ok ? "#34d399" : "#fbbf24", opacity: ok ? 0.85 : 0.95 }}
        strokeDasharray={ok ? "0" : "3 2"}
        transition={{ duration: 0.35 }}
      >
        <Shapes view={view} />
      </motion.g>
    </svg>
  );
}
