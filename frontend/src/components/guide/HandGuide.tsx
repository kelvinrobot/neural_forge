import { motion, useTransform, type MotionValue } from "framer-motion";
import type { GuideSide, HandPoseName } from "./stepAnimations";
import { keys, point, radians, shade, useLoopProgress, wave } from "./motion";

type HandFace = "palm" | "back";

interface Finger {
  baseX: number;
  length: number;
  angle: number;
}

const FINGERS: Finger[] = [
  { baseX: -20, length: 36, angle: -10 },
  { baseX: -7, length: 41, angle: -3 },
  { baseX: 7, length: 38, angle: 3 },
  { baseX: 20, length: 30, angle: 10 },
];

function fingerTip(finger: Finger, curl = 0): [number, number] {
  const angle = radians(finger.angle);
  const length = finger.length * (1 - curl * 0.6);
  return [finger.baseX + Math.sin(angle) * length, -56 - Math.cos(angle) * length];
}

interface StaticHandProps {
  x: number;
  y: number;
  scale?: number;
  mirrored?: boolean;
  face: HandFace;
  skin: string;
  curled?: boolean;
}

/** An upright hand drawn around its wrist point; fingers point up. */
function StaticHand({ x, y, scale = 1, mirrored = false, face, skin, curled = false }: StaticHandProps) {
  const outline = shade(skin, 0.75);
  return (
    <g transform={`translate(${x} ${y}) scale(${mirrored ? -scale : scale} ${scale})`}>
      {FINGERS.map((finger) => {
        const [tipX, tipY] = fingerTip(finger, curled ? 1 : 0);
        return (
          <g key={finger.baseX}>
            <path d={`M ${finger.baseX} -50 L ${point(tipX, tipY)}`} stroke={outline} strokeWidth="14.5" strokeLinecap="round" />
            <path d={`M ${finger.baseX} -50 L ${point(tipX, tipY)}`} stroke={skin} strokeWidth="12" strokeLinecap="round" />
            {face === "back" && (
              <rect x={tipX - 4} y={tipY - 2} width="8" height="10" rx="3.5" fill="#f4d6cf" stroke={outline} strokeWidth="0.8" />
            )}
          </g>
        );
      })}
      <path d="M -26 -22 L -44 -48" stroke={outline} strokeWidth="15" strokeLinecap="round" />
      <path d="M -26 -22 L -44 -48" stroke={skin} strokeWidth="12.5" strokeLinecap="round" />
      <rect x="-28" y="-60" width="56" height="60" rx="18" fill={skin} stroke={outline} strokeWidth="1.5" />
      <rect x="-15" y="-4" width="30" height="34" fill={skin} stroke={outline} strokeWidth="1.5" />
      {face === "palm" ? (
        <path d="M -18 -36 q 14 6 34 -4 M -16 -24 q 14 4 30 -2 M -6 -50 q 4 20 -2 40" fill="none" stroke={outline} strokeWidth="1.3" strokeLinecap="round" />
      ) : (
        <path d="M -20 -54 h4 M -7 -56 h4 M 6 -56 h4 M 19 -54 h4" stroke={outline} strokeWidth="1.3" strokeLinecap="round" />
      )}
    </g>
  );
}

interface HandGuideProps {
  pose: HandPoseName;
  side?: GuideSide;
  playing: boolean;
  skinTone: string;
}

const DURATIONS: Record<HandPoseName, number> = {
  palms: 3,
  backs: 3,
  side_view: 3,
  tap: 0.5,
  flip: 1.2,
  nail_baseline: 3,
  nail_press: 3,
  refill: 4,
  window: 3,
  lap: 4,
};

const HOLD_AT: Record<HandPoseName, number> = {
  palms: 0.25,
  backs: 0.25,
  side_view: 0.25,
  tap: 0,
  flip: 0,
  nail_baseline: 0.25,
  nail_press: 0.5,
  refill: 0.9,
  window: 0.25,
  lap: 0.25,
};

function useFloat(progress: MotionValue<number>, amplitude: number) {
  return useTransform(progress, (p) => wave(p, 1) * amplitude);
}

function TappingHand({ progress, mirrored, skin }: { progress: MotionValue<number>; mirrored: boolean; skin: string }) {
  const outline = shade(skin, 0.75);
  const opening = useTransform(progress, (p) => keys(p, [[0, 0], [0.5, 1], [1, 0]]));
  const indexPath = useTransform(opening, (o) => {
    const angle = radians(-30 - o * 15);
    return `M -20 -50 L ${point(-20 + Math.sin(angle) * 40, -50 - Math.cos(angle) * 40)}`;
  });
  const thumbPath = useTransform(opening, (o) => {
    const angle = radians(-80 + o * 30);
    return `M -26 -22 L ${point(-26 + Math.sin(angle) * 34, -22 - Math.cos(angle) * 34)}`;
  });
  return (
    <g transform={`translate(120 190) scale(${mirrored ? -1.3 : 1.3} 1.3)`}>
      {FINGERS.slice(1).map((finger) => {
        const [tipX, tipY] = fingerTip(finger, 1);
        return (
          <g key={finger.baseX}>
            <path d={`M ${finger.baseX} -50 L ${point(tipX, tipY)}`} stroke={outline} strokeWidth="14.5" strokeLinecap="round" />
            <path d={`M ${finger.baseX} -50 L ${point(tipX, tipY)}`} stroke={skin} strokeWidth="12" strokeLinecap="round" />
          </g>
        );
      })}
      <motion.path d={indexPath} stroke={outline} strokeWidth="14.5" strokeLinecap="round" />
      <motion.path d={indexPath} stroke={skin} strokeWidth="12" strokeLinecap="round" />
      <motion.path d={thumbPath} stroke={outline} strokeWidth="15" strokeLinecap="round" />
      <motion.path d={thumbPath} stroke={skin} strokeWidth="12.5" strokeLinecap="round" />
      <rect x="-28" y="-60" width="56" height="60" rx="18" fill={skin} stroke={outline} strokeWidth="1.5" />
      <rect x="-15" y="-4" width="30" height="34" fill={skin} stroke={outline} strokeWidth="1.5" />
    </g>
  );
}

function FlippingHand({ progress, skin }: { progress: MotionValue<number>; skin: string }) {
  const scaleX = useTransform(progress, (p) => Math.cos(p * Math.PI * 2));
  const palmOpacity = useTransform(scaleX, (value) => (value >= 0 ? 1 : 0));
  const backOpacity = useTransform(scaleX, (value) => (value < 0 ? 1 : 0));
  return (
    <>
      <motion.g style={{ scaleX }}>
        <motion.g style={{ opacity: palmOpacity }}>
          <StaticHand x={120} y={190} scale={1.3} face="palm" skin={skin} />
        </motion.g>
        <motion.g style={{ opacity: backOpacity }}>
          <StaticHand x={120} y={190} scale={1.3} face="back" skin={skin} />
        </motion.g>
      </motion.g>
      <path d="M 70 214 h 100" stroke="#46566b" strokeWidth="8" strokeLinecap="round" opacity="0.5" />
    </>
  );
}

function NailPress({ progress, skin, mode }: { progress: MotionValue<number>; skin: string; mode: "baseline" | "press" | "refill" }) {
  const outline = shade(skin, 0.75);
  const pressAmount = useTransform(progress, (p) => {
    if (mode === "baseline") return 0;
    if (mode === "press") return keys(p, [[0, 0], [0.25, 1], [0.9, 1], [1, 0]]);
    return keys(p, [[0, 1], [0.12, 0], [1, 0]]);
  });
  const nailColour = useTransform(progress, (p) => {
    if (mode === "baseline") return "#f2a7a3";
    if (mode === "press") return keys(p, [[0.2, 0], [0.35, 1], [0.9, 1], [1, 0]]) > 0.5 ? "#f6eeea" : "#f2a7a3";
    return p < 0.12 ? "#f6eeea" : p < 0.4 ? "#f3c9c3" : "#f2a7a3";
  });
  const thumbX = useTransform(pressAmount, (amount) => 200 - amount * 66);
  return (
    <>
      <StaticHand x={110} y={210} scale={1.45} face="back" skin={skin} />
      <motion.rect x="66" y="74" width="13" height="16" rx="5" style={{ fill: nailColour }} stroke={outline} strokeWidth="1" />
      {mode !== "baseline" && (
        <motion.g style={{ x: thumbX }}>
          <rect x="-70" y="72" width="70" height="20" rx="10" fill={skin} stroke={outline} strokeWidth="1.5" />
        </motion.g>
      )}
    </>
  );
}

function NailWindow({ skin }: { skin: string }) {
  const outline = shade(skin, 0.75);
  return (
    <g>
      <path d="M 70 40 Q 116 92 112 132 Q 108 150 96 150 L 60 60 Z" fill={skin} stroke={outline} strokeWidth="1.5" />
      <path d="M 170 40 Q 124 92 128 132 Q 132 150 144 150 L 180 60 Z" fill={skin} stroke={outline} strokeWidth="1.5" />
      <path d="M 104 112 q 6 -8 6 -20 M 136 112 q -6 -8 -6 -20" stroke="#f4d6cf" strokeWidth="6" strokeLinecap="round" fill="none" />
      <motion.path
        d="M 120 102 L 113 116 L 120 130 L 127 116 Z"
        fill="#ffffff"
        stroke="#14a8a4"
        strokeWidth="1.5"
        strokeDasharray="3 2"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
      <text x="120" y="182" textAnchor="middle" className="fill-current text-[11px]">
        nails back to back
      </text>
    </g>
  );
}

function SideFinger({ skin, side }: { skin: string; side: GuideSide }) {
  const outline = shade(skin, 0.75);
  const flip = side === "right" ? -1 : 1;
  return (
    <g transform={`translate(120 120) scale(${flip} 1)`}>
      <path d="M -90 -18 L 40 -18 Q 72 -18 76 0 Q 72 20 40 20 L -90 24 Z" fill={skin} stroke={outline} strokeWidth="1.5" />
      <path d="M 34 -18 Q 64 -20 74 -6" stroke="#f4d6cf" strokeWidth="7" strokeLinecap="round" fill="none" />
      <path d="M -20 -18 q 4 18 0 40 M 16 -18 q 3 18 0 38" stroke={outline} strokeWidth="1.2" fill="none" />
    </g>
  );
}

function Lap({ skin, progress }: { skin: string; progress: MotionValue<number> }) {
  const breathe = useFloat(progress, 1.2);
  return (
    <motion.g style={{ y: breathe }}>
      <rect x="30" y="120" width="80" height="110" rx="30" fill="#46566b" />
      <rect x="130" y="120" width="80" height="110" rx="30" fill="#46566b" />
      <StaticHand x={70} y={210} scale={0.9} face="back" skin={skin} curled />
      <StaticHand x={170} y={210} scale={0.9} face="back" skin={skin} curled mirrored />
    </motion.g>
  );
}

export function HandGuide({ pose, side, playing, skinTone }: HandGuideProps) {
  const progress = useLoopProgress(DURATIONS[pose], playing, HOLD_AT[pose]);
  const float = useFloat(progress, 2);
  const mirroredForSide = side === "left";
  return (
    <svg viewBox="0 0 240 240" className="h-full w-full text-ink-500" role="img" aria-label="Hand guide animation">
      {pose === "palms" && (
        <motion.g style={{ y: float }}>
          <StaticHand x={70} y={200} face="palm" skin={skinTone} mirrored />
          <StaticHand x={170} y={200} face="palm" skin={skinTone} />
        </motion.g>
      )}
      {pose === "backs" && (
        <motion.g style={{ y: float }}>
          <StaticHand x={70} y={200} face="back" skin={skinTone} />
          <StaticHand x={170} y={200} face="back" skin={skinTone} mirrored />
        </motion.g>
      )}
      {pose === "side_view" && (
        <motion.g style={{ y: float }}>
          <SideFinger skin={skinTone} side={side ?? "right"} />
        </motion.g>
      )}
      {pose === "tap" && <TappingHand progress={progress} mirrored={mirroredForSide} skin={skinTone} />}
      {pose === "flip" && <FlippingHand progress={progress} skin={skinTone} />}
      {pose === "nail_baseline" && <NailPress progress={progress} skin={skinTone} mode="baseline" />}
      {pose === "nail_press" && <NailPress progress={progress} skin={skinTone} mode="press" />}
      {pose === "refill" && <NailPress progress={progress} skin={skinTone} mode="refill" />}
      {pose === "window" && <NailWindow skin={skinTone} />}
      {pose === "lap" && <Lap skin={skinTone} progress={progress} />}
    </svg>
  );
}
