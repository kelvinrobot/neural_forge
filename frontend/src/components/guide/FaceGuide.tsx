import { motion, useTransform, type MotionValue } from "framer-motion";
import { useId } from "react";
import type { FacePoseName } from "./stepAnimations";
import { keys, point, shade, useLoopProgress, wave } from "./motion";

interface FaceParams {
  gazeX: number;
  gazeY: number;
  lid: number;
  squeeze: number;
  brow: number;
  browInner: number;
  open: number;
  smile: number;
  round: number;
  teeth: number;
  tongue: number;
  puff: number;
  tilt: number;
  zoom: number;
  lidPull: number;
  lipPull: number;
}

const NEUTRAL: FaceParams = {
  gazeX: 0,
  gazeY: 0,
  lid: 1,
  squeeze: 0,
  brow: 0,
  browInner: 0,
  open: 0,
  smile: 0.15,
  round: 0,
  teeth: 0,
  tongue: 0,
  puff: 0,
  tilt: 0,
  zoom: 1,
  lidPull: 0,
  lipPull: 0,
};

function blink(p: number, at: number): number {
  return 1 - keys(p, [
    [at - 0.04, 0],
    [at, 1],
    [at + 0.04, 0],
  ]);
}

const holdCurve = (p: number) =>
  keys(p, [
    [0, 0],
    [0.25, 1],
    [0.85, 1],
    [1, 0],
  ]);

interface FacePose {
  duration: number;
  holdAt: number;
  params: (p: number) => Partial<FaceParams>;
}

const POSES: Record<FacePoseName, FacePose> = {
  neutral: { duration: 4, holdAt: 0.3, params: (p) => ({ lid: blink(p, 0.85) }) },
  straight: { duration: 3.5, holdAt: 0.3, params: (p) => ({ lid: Math.min(blink(p, 0.45), blink(p, 0.93)) }) },
  smile: {
    duration: 3,
    holdAt: 0.5,
    params: (p) => {
      const amount = holdCurve(p);
      return { smile: 0.15 + amount * 0.85, teeth: amount > 0.5 ? 1 : 0, lid: 1 - amount * 0.15 };
    },
  },
  eyebrows: { duration: 3, holdAt: 0.5, params: (p) => ({ brow: holdCurve(p), lid: 1 + holdCurve(p) * 0.25 }) },
  eyes_closed: {
    duration: 3,
    holdAt: 0.5,
    params: (p) => {
      const amount = holdCurve(p);
      return { lid: 1 - amount, squeeze: amount, brow: -amount * 0.5 };
    },
  },
  cheeks: { duration: 3, holdAt: 0.5, params: (p) => ({ puff: holdCurve(p), smile: 0 }) },
  expressions: {
    duration: 7.5,
    holdAt: 0.18,
    params: (p) => {
      const happy = keys(p, [
        [0, 0],
        [0.07, 1],
        [0.26, 1],
        [0.33, 0],
      ]);
      const surprised = keys(p, [
        [0.33, 0],
        [0.4, 1],
        [0.6, 1],
        [0.67, 0],
      ]);
      const sad = keys(p, [
        [0.67, 0],
        [0.74, 1],
        [0.93, 1],
        [1, 0],
      ]);
      return {
        smile: 0.1 + happy * 0.9 - sad * 0.9,
        teeth: happy > 0.5 ? 1 : 0,
        brow: surprised,
        browInner: sad,
        open: surprised * 0.55,
        round: surprised,
        lid: 1 + surprised * 0.35 - sad * 0.2,
        gazeY: sad * 0.4,
      };
    },
  },
  talking: {
    duration: 3,
    holdAt: 0.1,
    params: (p) => ({ open: 0.08 + 0.28 * Math.abs(wave(p, 7)), lid: blink(p, 0.7), smile: 0.2 }),
  },
  gaze_left: { duration: 3.5, holdAt: 0.5, params: (p) => ({ gazeX: holdCurve(p) }) },
  gaze_right: { duration: 3.5, holdAt: 0.5, params: (p) => ({ gazeX: -holdCurve(p) }) },
  gaze_up: { duration: 3.5, holdAt: 0.5, params: (p) => ({ gazeY: -holdCurve(p), lid: 1 + holdCurve(p) * 0.15 }) },
  gaze_down: { duration: 3.5, holdAt: 0.5, params: (p) => ({ gazeY: holdCurve(p), lid: 1 - holdCurve(p) * 0.35 }) },
  upgaze: { duration: 5, holdAt: 0.5, params: (p) => ({ gazeY: -holdCurve(p), lid: 1 + holdCurve(p) * 0.15, brow: holdCurve(p) * 0.3 }) },
  close_up: {
    duration: 4,
    holdAt: 0.7,
    params: (p) => {
      const zoom = keys(p, [
        [0, 0],
        [0.5, 1],
        [0.9, 1],
        [1, 0],
      ]);
      return { zoom: 1 + zoom * 0.35, lid: 1 + zoom * 0.3 };
    },
  },
  lower_lid: { duration: 4, holdAt: 0.5, params: (p) => ({ lidPull: holdCurve(p), gazeY: -holdCurve(p) * 0.8 }) },
  mouth_closed: { duration: 4, holdAt: 0.3, params: (p) => ({ lid: blink(p, 0.8), smile: 0.05 }) },
  open_wide: { duration: 3, holdAt: 0.5, params: (p) => ({ open: holdCurve(p), smile: 0, lid: 1 + holdCurve(p) * 0.1 }) },
  tongue_out: {
    duration: 3,
    holdAt: 0.5,
    params: (p) => ({ open: 0.5 * holdCurve(p), tongue: holdCurve(p), smile: 0 }),
  },
  aaah: {
    duration: 3,
    holdAt: 0.5,
    params: (p) => {
      const amount = holdCurve(p);
      return { open: amount * (0.9 + 0.05 * wave(p, 12)), tilt: -9 * amount, gazeY: -0.3 * amount, smile: 0, tongue: 0.1 * amount };
    },
  },
  gums: {
    duration: 3.5,
    holdAt: 0.5,
    params: (p) => {
      const amount = holdCurve(p);
      return { lipPull: amount, teeth: amount > 0.2 ? 1 : 0, smile: 0.3 + 0.4 * amount, open: 0.25 * amount };
    },
  },
};

const EYES = [
  { x: 76, side: "right" },
  { x: 124, side: "left" },
] as const;
const EYE_Y = 108;

function mouthPath(face: FaceParams): string {
  const width = 22 + face.smile * 6 - face.round * 10 - face.puff * 7 + face.lipPull * 9;
  const cornerY = 160 - face.smile * 7 - face.lipPull * 2;
  const upper = Math.min(157 - face.open * 6 - face.round * 3, cornerY - 2);
  const lower = Math.max(162 + face.open * 32 + Math.max(0, face.smile) * 14 + face.lipPull * 8, cornerY + 2);
  return `M ${point(100 - width, cornerY)} Q ${point(100, upper)} ${point(100 + width, cornerY)} Q ${point(100, lower)} ${point(100 - width, cornerY)} Z`;
}

function browPath(face: FaceParams, eyeX: number, side: "right" | "left"): string {
  const inward = side === "right" ? 1 : -1;
  const base = 90 - face.brow * 8 + face.squeeze * 3;
  const innerY = base - face.browInner * 6 + Math.max(0, -face.brow) * 5;
  const outerY = base + face.browInner * 2;
  const innerX = eyeX + inward * 13;
  const outerX = eyeX - inward * 14;
  return `M ${point(outerX, outerY)} Q ${point(eyeX, base - 4)} ${point(innerX, innerY)}`;
}

function useFace(progress: MotionValue<number>, pose: FacePose) {
  return useTransform(progress, (p) => ({ ...NEUTRAL, ...pose.params(p) }));
}

interface FaceGuideProps {
  pose: FacePoseName;
  playing: boolean;
  skinTone: string;
}

export function FaceGuide({ pose, playing, skinTone }: FaceGuideProps) {
  const definition = POSES[pose];
  const progress = useLoopProgress(definition.duration, playing, definition.holdAt);
  const face = useFace(progress, definition);
  const clipId = useId();
  const outline = shade(skinTone, 0.78);

  const headRx = useTransform(face, (f) => 60 + f.puff * 6);
  const cheekR = useTransform(face, (f) => 10 + f.puff * 9);
  const cheekOpacity = useTransform(face, (f) => 0.18 + f.puff * 0.35 + Math.max(0, f.smile) * 0.1);
  const eyeRy = useTransform(face, (f) => Math.max(0.6, 8 * Math.min(f.lid, 1.4)));
  const irisOpacity = useTransform(face, (f) => (f.lid > 0.3 ? 1 : 0));
  const closedOpacity = useTransform(face, (f) => (f.lid <= 0.3 ? 1 : 0));
  const squeezeOpacity = useTransform(face, (f) => f.squeeze);
  const irisRightX = useTransform(face, (f) => EYES[0].x + f.gazeX * 5.5);
  const irisLeftX = useTransform(face, (f) => EYES[1].x + f.gazeX * 5.5);
  const irisY = useTransform(face, (f) => EYE_Y + f.gazeY * 3.4);
  const browRight = useTransform(face, (f) => browPath(f, EYES[0].x, "right"));
  const browLeft = useTransform(face, (f) => browPath(f, EYES[1].x, "left"));
  const mouth = useTransform(face, mouthPath);
  const teethOpacity = useTransform(face, (f) => f.teeth);
  const gumOpacity = useTransform(face, (f) => f.lipPull);
  const tongueOutY = useTransform(face, (f) => 172 + f.open * 20 + f.tongue * 14);
  const tongueOutRy = useTransform(face, (f) => 6 + f.tongue * 13);
  const tongueOutOpacity = useTransform(face, (f) => Math.min(1, Math.max(0, (f.tongue - 0.25) / 0.3)));
  const innerTongueY = useTransform(face, (f) => 166 + f.open * 26);
  const groupRotate = useTransform(face, (f) => f.tilt);
  const groupScale = useTransform(face, (f) => f.zoom);
  const lidFingerY = useTransform(face, (f) => 132 - f.lidPull * 8);
  const lidPullOpacity = useTransform(face, (f) => (f.lidPull > 0.05 ? 1 : 0));
  const conjunctivaOpacity = useTransform(face, (f) => f.lidPull);
  const lipFingerRightX = useTransform(face, (f) => 64 - f.lipPull * 8);
  const lipFingerLeftX = useTransform(face, (f) => 136 + f.lipPull * 8);

  return (
    <svg viewBox="0 0 200 230" className="h-full w-full" role="img" aria-label="Face guide animation">
      <motion.g style={{ rotate: groupRotate, scale: groupScale }}>
        <path d="M58 196c8-14 26-20 42-20s34 6 42 20l22 34H36z" fill="#6aa9a6" />
        <rect x="86" y="168" width="28" height="24" rx="10" fill={skinTone} />
        <ellipse cx="38" cy="114" rx="9" ry="14" fill={skinTone} stroke={outline} strokeWidth="1.5" />
        <ellipse cx="162" cy="114" rx="9" ry="14" fill={skinTone} stroke={outline} strokeWidth="1.5" />
        <motion.ellipse cx="100" cy="112" rx={headRx} ry="72" fill={skinTone} stroke={outline} strokeWidth="1.5" />
        <path d="M40 104c0-44 26-66 60-66s60 22 60 66c-6-18-18-30-34-36-14 10-44 14-72 8-6 8-12 18-14 28z" fill="#3b3027" />
        <motion.circle cx="64" cy="140" r={cheekR} fill="#e7867f" style={{ opacity: cheekOpacity }} />
        <motion.circle cx="136" cy="140" r={cheekR} fill="#e7867f" style={{ opacity: cheekOpacity }} />

        {EYES.map((eye, index) => (
          <g key={eye.side}>
            <motion.ellipse cx={eye.x} cy={EYE_Y} rx="13" ry={eyeRy} fill="#ffffff" stroke={outline} strokeWidth="1.2" />
            <motion.circle
              cx={index === 0 ? irisRightX : irisLeftX}
              cy={irisY}
              r="5.4"
              fill="#4a6b7c"
              style={{ opacity: irisOpacity }}
            />
            <motion.circle
              cx={index === 0 ? irisRightX : irisLeftX}
              cy={irisY}
              r="2.4"
              fill="#16222b"
              style={{ opacity: irisOpacity }}
            />
            <motion.path
              d={`M ${eye.x - 13} ${EYE_Y} Q ${eye.x} ${EYE_Y + 6} ${eye.x + 13} ${EYE_Y}`}
              stroke="#3b3027"
              strokeWidth="2"
              fill="none"
              strokeLinecap="round"
              style={{ opacity: closedOpacity }}
            />
            <motion.path
              d={`M ${eye.x - 17} ${EYE_Y - 6} l 5 3 M ${eye.x + 17} ${EYE_Y - 6} l -5 3 M ${eye.x - 16} ${EYE_Y + 7} l 5 -2`}
              stroke={outline}
              strokeWidth="1.5"
              strokeLinecap="round"
              style={{ opacity: squeezeOpacity }}
            />
            <motion.path
              d={`M ${eye.x - 12} ${EYE_Y + 9} Q ${eye.x} ${EYE_Y + 17} ${eye.x + 12} ${EYE_Y + 9} Q ${eye.x} ${EYE_Y + 12} ${eye.x - 12} ${EYE_Y + 9} Z`}
              fill="#e57f88"
              style={{ opacity: conjunctivaOpacity }}
            />
          </g>
        ))}

        <motion.path d={browRight} stroke="#3b3027" strokeWidth="4" fill="none" strokeLinecap="round" />
        <motion.path d={browLeft} stroke="#3b3027" strokeWidth="4" fill="none" strokeLinecap="round" />
        <path d="M100 116c-3 10-6 16-6 20 0 3 3 4 6 4s6-1 6-4" fill="none" stroke={outline} strokeWidth="2" strokeLinecap="round" />

        <defs>
          <clipPath id={clipId}>
            <motion.path d={mouth} />
          </clipPath>
        </defs>
        <motion.path d={mouth} fill="#5a1f2a" />
        <g clipPath={`url(#${clipId})`}>
          <motion.rect x="60" y="146" width="80" height="10" fill="#e9919a" style={{ opacity: gumOpacity }} />
          <motion.rect x="60" y="148" width="80" height="20" fill="#ffffff" style={{ opacity: teethOpacity }} />
          <motion.ellipse cx="100" cy={innerTongueY} rx="15" ry="9" fill="#e07c86" />
        </g>
        <motion.path d={mouth} fill="none" stroke="#b85c62" strokeWidth="3" strokeLinejoin="round" />
        <motion.ellipse cx="100" cy={tongueOutY} rx="13" ry={tongueOutRy} fill="#e07c86" stroke="#c8606b" strokeWidth="1.5" style={{ opacity: tongueOutOpacity }} />

        <motion.g style={{ opacity: lidPullOpacity }}>
          {EYES.map((eye) => (
            <motion.rect key={eye.side} x={eye.x - 6} y={lidFingerY} width="12" height="40" rx="6" fill={skinTone} stroke={outline} strokeWidth="1.5" />
          ))}
        </motion.g>
        <motion.g style={{ opacity: gumOpacity }}>
          <motion.rect x={lipFingerRightX} y="154" width="30" height="12" rx="6" fill={skinTone} stroke={outline} strokeWidth="1.5" transform="translate(-30 0)" />
          <motion.rect x={lipFingerLeftX} y="154" width="30" height="12" rx="6" fill={skinTone} stroke={outline} strokeWidth="1.5" />
        </motion.g>
      </motion.g>
    </svg>
  );
}
