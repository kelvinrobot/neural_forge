import { motion, useTransform } from "framer-motion";
import type { GuideSide, SkinPoseName } from "./stepAnimations";
import { keys, shade, useLoopProgress, wave } from "./motion";

interface SkinGuideProps {
  pose: SkinPoseName;
  side?: GuideSide;
  playing: boolean;
  skinTone: string;
}

function TurnHead({ side, skinTone, playing }: { side: GuideSide; skinTone: string; playing: boolean }) {
  const progress = useLoopProgress(3.5, playing, 0.5);
  // Showing the left cheek means turning toward the patient's right, which is the screen's left in a mirror.
  const direction = side === "left" ? -1 : 1;
  const turn = useTransform(progress, (p) =>
    keys(p, [
      [0, 0],
      [0.3, 1],
      [0.85, 1],
      [1, 0],
    ]),
  );
  const faceShift = useTransform(turn, (t) => direction * t * 18);
  const squeeze = useTransform(turn, (t) => 1 - t * 0.18);
  const earX = useTransform(turn, (t) => 100 - direction * (46 - t * 30));
  const outline = shade(skinTone, 0.75);
  return (
    <svg viewBox="0 0 200 230" className="h-full w-full text-ink-500" role="img" aria-label="Head turn guide animation">
      <path d="M58 200c8-14 26-20 42-20s34 6 42 20l22 30H36z" fill="#6aa9a6" />
      <rect x="86" y="164" width="28" height="26" rx="10" fill={skinTone} />
      <motion.g style={{ scaleX: squeeze }}>
        <ellipse cx="100" cy="108" rx="58" ry="70" fill={skinTone} stroke={outline} strokeWidth="1.5" />
        <path d="M42 100c0-42 24-62 58-62s58 20 58 62c-8-18-30-32-58-30s-50 12-58 30z" fill="#3b3027" />
      </motion.g>
      <motion.ellipse cx={earX} cy="112" rx="8" ry="14" fill={skinTone} stroke={outline} strokeWidth="1.5" />
      <motion.g style={{ x: faceShift }}>
        <circle cx="80" cy="104" r="3.5" fill="#16222b" />
        <circle cx="120" cy="104" r="3.5" fill="#16222b" />
        <path d="M100 110c-3 10-6 16-6 20 0 3 3 4 6 4" fill="none" stroke={outline} strokeWidth="2" strokeLinecap="round" />
        <path d="M86 152q14 8 28 0" fill="none" stroke="#b85c62" strokeWidth="3" strokeLinecap="round" />
      </motion.g>
      <motion.path
        d={direction < 0 ? "M150 40 q-50 -26 -100 0 m 0 0 l 10 -2 m -10 2 l 4 9" : "M50 40 q50 -26 100 0 m 0 0 l -10 -2 m 10 2 l -4 9"}
        stroke="#14a8a4"
        strokeWidth="3"
        fill="none"
        strokeLinecap="round"
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
    </svg>
  );
}

function Forearms({ inner, skinTone, playing }: { inner: boolean; skinTone: string; playing: boolean }) {
  const progress = useLoopProgress(3, playing, 0.25);
  const float = useTransform(progress, (p) => wave(p, 1) * 2);
  const tone = inner ? shade(skinTone, 1.06) : shade(skinTone, 0.92);
  const outline = shade(skinTone, 0.75);
  return (
    <svg viewBox="0 0 240 240" className="h-full w-full text-ink-500" role="img" aria-label="Forearm guide animation">
      <motion.g style={{ y: float }}>
        {[80, 160].map((x) => (
          <g key={x}>
            <rect x={x - 18} y="70" width="36" height="160" rx="18" fill={tone} stroke={outline} strokeWidth="1.5" />
            <rect x={x - 22} y="28" width="44" height="52" rx="16" fill={tone} stroke={outline} strokeWidth="1.5" />
            {inner ? (
              <path d={`M ${x - 12} 86 h 24 M ${x - 10} 92 h 20`} stroke={outline} strokeWidth="1.2" strokeLinecap="round" />
            ) : (
              <path
                d={`M ${x - 8} 120 l 3 -5 M ${x + 4} 140 l 3 -5 M ${x - 6} 165 l 3 -5 M ${x + 6} 190 l 3 -5`}
                stroke={outline}
                strokeWidth="1"
                strokeLinecap="round"
              />
            )}
          </g>
        ))}
      </motion.g>
      <text x="120" y="20" textAnchor="middle" className="fill-current text-[12px] font-medium">
        {inner ? "Inner side facing the camera" : "Outer side facing the camera"}
      </text>
    </svg>
  );
}

function AreaClose({ skinTone, playing }: { skinTone: string; playing: boolean }) {
  const progress = useLoopProgress(3.5, playing, 0.6);
  const scale = useTransform(progress, (p) =>
    keys(p, [
      [0, 0.7],
      [0.45, 1.05],
      [0.9, 1.05],
      [1, 0.7],
    ]),
  );
  return (
    <svg viewBox="0 0 240 240" className="h-full w-full text-ink-500" role="img" aria-label="Skin area guide animation">
      <motion.g style={{ scale }}>
        <rect x="70" y="40" width="100" height="100" rx="24" fill={skinTone} stroke={shade(skinTone, 0.75)} strokeWidth="1.5" />
        <circle cx="105" cy="80" r="9" fill={shade(skinTone, 0.7)} />
        <circle cx="132" cy="104" r="5" fill={shade(skinTone, 0.78)} />
        <rect x="64" y="34" width="112" height="112" rx="28" fill="none" stroke="#14a8a4" strokeWidth="2" strokeDasharray="6 5" />
      </motion.g>
      <g transform="translate(120 205)" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round">
        <rect x="-16" y="-10" width="32" height="20" rx="4" />
        <circle cx="0" cy="0" r="5" />
      </g>
      <path d="M 120 182 V 154" stroke="currentColor" strokeDasharray="4 3" strokeWidth="1.5" />
      <text x="160" y="172" className="fill-current text-[11px]">
        20–30 cm
      </text>
    </svg>
  );
}

export function SkinGuide({ pose, side = "right", playing, skinTone }: SkinGuideProps) {
  if (pose === "turn_head") return <TurnHead side={side} skinTone={skinTone} playing={playing} />;
  if (pose === "forearms_inner") return <Forearms inner skinTone={skinTone} playing={playing} />;
  if (pose === "forearms_outer") return <Forearms inner={false} skinTone={skinTone} playing={playing} />;
  return <AreaClose skinTone={skinTone} playing={playing} />;
}
