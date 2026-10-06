import { motion, useTransform } from "framer-motion";
import type { BodyPoseName, GuideSide } from "./stepAnimations";
import { keys, point, radians, shade, useLoopProgress, wave } from "./motion";

type BodyView = "front" | "side" | "back";

/**
 * Front/back views: arm = [abduction, elbow bend] and leg = [hip angle, shin scale] (shin scale < 1 lifts the foot).
 * Side view (figure faces `facing`): arm = [forward flexion, elbow bend] and leg = [hip flexion, knee flexion].
 * Angles are degrees measured from hanging straight down.
 */
interface BodyPose {
  view: BodyView;
  x: number;
  drop: number;
  lean: number;
  facing: 1 | -1;
  armRight: [number, number];
  armLeft: [number, number];
  legRight: [number, number];
  legLeft: [number, number];
  thighScale: number;
  stance: number;
  eyesClosed: boolean;
}

const BASE: BodyPose = {
  view: "front",
  x: 0,
  drop: 0,
  lean: 0,
  facing: 1,
  armRight: [8, 6],
  armLeft: [8, 6],
  legRight: [3, 1],
  legLeft: [3, 1],
  thighScale: 1,
  stance: 10,
  eyesClosed: false,
};

const CENTER_X = 120;
const HIP_Y = 186;
const SEAT_DROP = 52;
const TORSO = 72;
const NECK = 27;
const UPPER_ARM = 34;
const FOREARM = 30;
const THIGH = 52;
const SHIN = 52;

const SEATED: Partial<BodyPose> = { view: "side", drop: SEAT_DROP, legRight: [90, 90], legLeft: [90, 90] };

interface PoseDefinition {
  duration: number;
  holdAt: number;
  chair?: "behind" | "front" | "left";
  mark?: boolean;
  bubble?: boolean;
  caption?: string;
  pose: (p: number, side: GuideSide) => Partial<BodyPose>;
}

const POSES: Record<BodyPoseName, PoseDefinition> = {
  front: { duration: 4, holdAt: 0.25, pose: (p) => ({ drop: wave(p, 1) * 0.8 }) },
  back: { duration: 4, holdAt: 0.25, pose: (p) => ({ view: "back", drop: wave(p, 1) * 0.8 }) },
  side: { duration: 4, holdAt: 0.25, pose: (p) => ({ view: "side", drop: wave(p, 1) * 0.8, armRight: [4, 8], armLeft: [4, 8] }) },
  arms_side: {
    duration: 3.5,
    holdAt: 0.5,
    pose: (p) => {
      const angle = keys(p, [
        [0, 8],
        [0.45, 172],
        [0.55, 172],
        [1, 8],
      ]);
      return { armRight: [angle, 0], armLeft: [angle, 0] };
    },
  },
  knee_bends: {
    duration: 3.2,
    holdAt: 0.5,
    chair: "front",
    pose: (p) => {
      const bend = keys(p, [
        [0, 0],
        [0.5, 1],
        [1, 0],
      ]);
      return {
        view: "side",
        drop: bend * 34,
        lean: bend * 22,
        legRight: [bend * 70, bend * 110],
        legLeft: [bend * 70, bend * 110],
        armRight: [62, 10],
        armLeft: [58, 10],
      };
    },
  },
  balance_open: { duration: 4, holdAt: 0.25, pose: (p) => ({ stance: 4, lean: wave(p, 1) * 1.5, armRight: [4, 4], armLeft: [4, 4] }) },
  balance_closed: {
    duration: 4,
    holdAt: 0.25,
    pose: (p) => ({ stance: 4, lean: wave(p, 1) * 2.5, armRight: [4, 4], armLeft: [4, 4], eyesClosed: true }),
  },
  single_leg: {
    duration: 4,
    holdAt: 0.5,
    pose: (p, side) => {
      const lift = keys(p, [
        [0, 1],
        [0.25, 0.5],
        [0.85, 0.5],
        [1, 1],
      ]);
      const raised: [number, number] = [6, lift];
      return side === "right"
        ? { legLeft: raised, armRight: [18, 6], armLeft: [18, 6], lean: wave(p, 2) * 1.2 }
        : { legRight: raised, armRight: [18, 6], armLeft: [18, 6], lean: wave(p, 2) * 1.2 };
    },
  },
  walking: {
    duration: 4.8,
    holdAt: 0.5,
    pose: (p) => {
      const stride = wave(p, 4);
      return {
        view: "side",
        x: -80 + 160 * p,
        drop: Math.abs(stride) * -2,
        legRight: [stride * 24, Math.max(0, stride) * 30],
        legLeft: [-stride * 24, Math.max(0, -stride) * 30],
        armRight: [-stride * 22, 12],
        armLeft: [stride * 22, 12],
      };
    },
  },
  tug: {
    duration: 11,
    holdAt: 0,
    chair: "left",
    mark: true,
    pose: (p) => {
      const sit = keys(p, [
        [0, 1],
        [0.1, 1],
        [0.17, 0],
        [0.84, 0],
        [0.92, 1],
        [1, 1],
      ]);
      const outward = keys(p, [
        [0.17, -78],
        [0.48, 66],
        [0.55, 66],
        [0.84, -78],
      ]);
      const walking = p > 0.17 && p < 0.84;
      const stride = walking ? wave(p, 14) : 0;
      return {
        view: "side",
        facing: p > 0.5 && p < 0.88 ? -1 : 1,
        x: outward,
        drop: sit * SEAT_DROP,
        lean: sit * 8,
        legRight: [sit * 90 + stride * 22 * (1 - sit), sit * 90 + Math.max(0, stride) * 28],
        legLeft: [sit * 90 - stride * 22 * (1 - sit), sit * 90 + Math.max(0, -stride) * 28],
        armRight: [-stride * 18, 12],
        armLeft: [stride * 18, 12],
      };
    },
  },
  sit_to_stand: {
    duration: 2.6,
    holdAt: 0,
    chair: "behind",
    pose: (p) => {
      const sit = keys(p, [
        [0, 1],
        [0.4, 0],
        [0.6, 0],
        [1, 1],
      ]);
      return {
        view: "side",
        drop: sit * SEAT_DROP,
        lean: keys(p, [
          [0, 10],
          [0.2, 30],
          [0.4, 0],
          [0.6, 0],
          [0.8, 25],
          [1, 10],
        ]),
        legRight: [sit * 90, sit * 90],
        legLeft: [sit * 90, sit * 90],
        armRight: [30, 120],
        armLeft: [28, 118],
      };
    },
  },
  rest: {
    duration: 6,
    holdAt: 0.2,
    chair: "behind",
    bubble: true,
    pose: (p) => ({
      ...SEATED,
      view: "front",
      thighScale: 0.3,
      legRight: [4, 0.72],
      legLeft: [4, 0.72],
      armRight: [6, -38],
      armLeft: [6, -38],
      drop: SEAT_DROP + wave(p, 2) * 0.5,
    }),
  },
  postural: {
    duration: 4,
    holdAt: 0.5,
    chair: "behind",
    caption: "Palms down",
    pose: (p) => ({ ...SEATED, armRight: [88 + wave(p, 1) * 1, 0], armLeft: [86, 0] }),
  },
  drift: {
    duration: 4,
    holdAt: 0.6,
    chair: "behind",
    caption: "Palms up, eyes closed",
    pose: (p) => ({ ...SEATED, armRight: [88, 0], armLeft: [86, 0], eyesClosed: p > 0.2 }),
  },
  nose: {
    duration: 2.4,
    holdAt: 0,
    chair: "behind",
    pose: (p) => {
      const reach = keys(p, [
        [0, 0],
        [0.4, 1],
        [0.55, 1],
        [0.95, 0],
        [1, 0],
      ]);
      return { ...SEATED, armRight: [90 + reach * 12, reach * 152], armLeft: [8, 30] };
    },
  },
};

interface Geometry {
  head: [number, number];
  torso: string;
  armRight: string;
  armLeft: string;
  handRight: [number, number];
  handLeft: [number, number];
  legRight: string;
  legLeft: string;
  eyesClosed: boolean;
  view: BodyView;
  facing: 1 | -1;
}

function limb(start: [number, number], directionX: (angle: number) => number, first: number, second: number, length1: number, length2: number) {
  const elbow: [number, number] = [start[0] + directionX(first) * length1, start[1] + Math.cos(radians(first)) * length1];
  const end: [number, number] = [elbow[0] + directionX(second) * length2, elbow[1] + Math.cos(radians(second)) * length2];
  return { path: `M ${point(...start)} L ${point(...elbow)} L ${point(...end)}`, end };
}

function geometry(pose: BodyPose): Geometry {
  const hip: [number, number] = [CENTER_X + pose.x, HIP_Y + pose.drop];
  if (pose.view === "side") {
    const facing = pose.facing;
    const forward = (angle: number) => facing * Math.sin(radians(angle));
    const shoulder: [number, number] = [hip[0] + forward(pose.lean) * TORSO, hip[1] - Math.cos(radians(pose.lean)) * TORSO];
    const head: [number, number] = [shoulder[0] + forward(pose.lean) * NECK + facing * 2, shoulder[1] - Math.cos(radians(pose.lean)) * NECK];
    const armRight = limb(shoulder, forward, pose.armRight[0], pose.armRight[0] + pose.armRight[1], UPPER_ARM, FOREARM);
    const armLeft = limb(shoulder, forward, pose.armLeft[0], pose.armLeft[0] + pose.armLeft[1], UPPER_ARM, FOREARM);
    const legRight = limb(hip, forward, pose.legRight[0], pose.legRight[0] - pose.legRight[1], THIGH, SHIN);
    const legLeft = limb(hip, forward, pose.legLeft[0], pose.legLeft[0] - pose.legLeft[1], THIGH, SHIN);
    return {
      head,
      torso: `M ${point(...hip)} L ${point(...shoulder)}`,
      armRight: armRight.path,
      armLeft: armLeft.path,
      handRight: armRight.end,
      handLeft: armLeft.end,
      legRight: `${legRight.path} l ${facing * 10} 0`,
      legLeft: `${legLeft.path} l ${facing * 10} 0`,
      eyesClosed: pose.eyesClosed,
      view: "side",
      facing,
    };
  }
  // Mirror view: the patient's right side is drawn on the screen's left; from behind it is on the right.
  const rightSign = pose.view === "front" ? -1 : 1;
  const tilt = radians(pose.lean);
  const shoulderCenter: [number, number] = [hip[0] + Math.sin(tilt) * TORSO, hip[1] - Math.cos(tilt) * TORSO];
  const head: [number, number] = [shoulderCenter[0] + Math.sin(tilt) * NECK, shoulderCenter[1] - Math.cos(tilt) * NECK];
  const sidePoint = (origin: [number, number], sign: number, offset: number): [number, number] => [origin[0] + sign * offset, origin[1]];
  const outward = (sign: number) => (angle: number) => sign * Math.sin(radians(angle));
  const build = (sign: number, arm: [number, number], leg: [number, number]) => {
    const armLimb = limb(sidePoint(shoulderCenter, sign, 21), outward(sign), arm[0], arm[0] + arm[1], UPPER_ARM, FOREARM);
    const legLimb = limb(sidePoint(hip, sign, pose.stance), outward(sign), leg[0], leg[0] * 0.5, THIGH * pose.thighScale, SHIN * leg[1]);
    return { arm: armLimb, leg: legLimb };
  };
  const right = build(rightSign, pose.armRight, pose.legRight);
  const left = build(-rightSign, pose.armLeft, pose.legLeft);
  return {
    head,
    torso: `M ${point(...hip)} L ${point(...shoulderCenter)}`,
    armRight: right.arm.path,
    armLeft: left.arm.path,
    handRight: right.arm.end,
    handLeft: left.arm.end,
    legRight: `${right.leg.path} l ${rightSign * 6} 2`,
    legLeft: `${left.leg.path} l ${-rightSign * 6} 2`,
    eyesClosed: pose.eyesClosed,
    view: pose.view,
    facing: 1,
  };
}

const SHIRT = "#6aa9a6";
const TROUSERS = "#46566b";
const HAIR = "#3b3027";

interface BodyGuideProps {
  pose: BodyPoseName;
  side?: GuideSide;
  playing: boolean;
  skinTone: string;
}

export function BodyGuide({ pose, side = "right", playing, skinTone }: BodyGuideProps) {
  const definition = POSES[pose];
  const progress = useLoopProgress(definition.duration, playing, definition.holdAt);
  const geo = useTransform(progress, (p) => geometry({ ...BASE, ...definition.pose(p, side) }));
  const outline = shade(skinTone, 0.75);

  const torso = useTransform(geo, (g) => g.torso);
  const torsoWidth = useTransform(geo, (g) => (g.view === "side" ? 24 : 40));
  const armRight = useTransform(geo, (g) => g.armRight);
  const armLeft = useTransform(geo, (g) => g.armLeft);
  const legRight = useTransform(geo, (g) => g.legRight);
  const legLeft = useTransform(geo, (g) => g.legLeft);
  const handRightX = useTransform(geo, (g) => g.handRight[0]);
  const handRightY = useTransform(geo, (g) => g.handRight[1]);
  const handLeftX = useTransform(geo, (g) => g.handLeft[0]);
  const handLeftY = useTransform(geo, (g) => g.handLeft[1]);
  const headX = useTransform(geo, (g) => g.head[0]);
  const headY = useTransform(geo, (g) => g.head[1]);
  const hairPath = useTransform(geo, (g) => {
    const [x, y] = g.head;
    if (g.view === "back") return `M ${point(x - 16, y)} a 16 16 0 1 1 32 0 a 16 17 0 0 1 -32 0 Z`;
    if (g.view === "side") return `M ${point(x - g.facing * 16, y + 4)} a 16 16 0 0 1 ${g.facing * 28} -12 q -10 2 -18 10 Z`;
    return `M ${point(x - 16, y - 2)} a 16 16 0 0 1 32 0 q -14 -6 -32 0 Z`;
  });
  const faceVisible = useTransform(geo, (g) => (g.view === "back" ? 0 : 1));
  const openEyes = useTransform(geo, (g) => (g.eyesClosed ? 0 : 1));
  const closedEyes = useTransform(geo, (g) => (g.eyesClosed ? 1 : 0));
  const eyesPath = useTransform(geo, (g) => {
    const [x, y] = g.head;
    if (g.view === "side") return `M ${point(x + g.facing * 7, y - 2)} h ${g.facing * 0.1}`;
    return `M ${point(x - 6, y - 1)} h 0.1 M ${point(x + 6, y - 1)} h 0.1`;
  });
  const closedPath = useTransform(geo, (g) => {
    const [x, y] = g.head;
    if (g.view === "side") return `M ${point(x + g.facing * 4, y - 1)} q ${g.facing * 3} 2 ${g.facing * 6} 0`;
    return `M ${point(x - 9, y - 1)} q 3 2 6 0 M ${point(x + 3, y - 1)} q 3 2 6 0`;
  });
  const bubbleText = useTransform(progress, (p) => `${20 - Math.floor(p * 6)}…`);

  return (
    <svg viewBox="0 0 240 300" className="h-full w-full text-ink-500" role="img" aria-label="Body guide animation">
      <line x1="10" y1="292" x2="230" y2="292" stroke="currentColor" strokeOpacity="0.35" strokeWidth="2" />
      {definition.chair === "behind" && <Chair x={pose === "rest" ? 120 : 104} front={pose === "rest"} />}
      {definition.chair === "left" && <Chair x={30} front={false} />}
      {definition.chair === "front" && <Chair x={182} front={false} />}
      {definition.mark && (
        <g>
          <path d="M 196 284 l 12 12 M 208 284 l -12 12" stroke="#d9480f" strokeWidth="3" strokeLinecap="round" />
          <text x="202" y="276" textAnchor="middle" className="fill-current text-[10px]">
            3 m
          </text>
        </g>
      )}

      <motion.path d={legLeft} stroke={shade(TROUSERS, 0.82)} strokeWidth="13" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <motion.path d={armLeft} stroke={shade(SHIRT, 0.82)} strokeWidth="10.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <motion.circle cx={handLeftX} cy={handLeftY} r="5.5" fill={shade(skinTone, 0.9)} stroke={outline} strokeWidth="1" />
      <motion.path d={torso} stroke={SHIRT} strokeWidth={torsoWidth} strokeLinecap="round" fill="none" />
      <motion.path d={legRight} stroke={TROUSERS} strokeWidth="13" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <motion.circle cx={headX} cy={headY} r="16" fill={skinTone} stroke={outline} strokeWidth="1.5" />
      <motion.path d={hairPath} fill={HAIR} />
      <motion.g style={{ opacity: faceVisible }}>
        <motion.path d={eyesPath} stroke="#16222b" strokeWidth="3.2" strokeLinecap="round" style={{ opacity: openEyes }} />
        <motion.path d={closedPath} stroke="#16222b" strokeWidth="1.6" strokeLinecap="round" fill="none" style={{ opacity: closedEyes }} />
      </motion.g>
      <motion.path d={armRight} stroke={SHIRT} strokeWidth="10.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <motion.circle cx={handRightX} cy={handRightY} r="5.5" fill={skinTone} stroke={outline} strokeWidth="1" />

      {definition.bubble && (
        <g>
          <path d="M 150 40 h 64 a 10 10 0 0 1 10 10 v 18 a 10 10 0 0 1 -10 10 h -48 l -10 10 v -10 h -6 a 10 10 0 0 1 -10 -10 v -18 a 10 10 0 0 1 10 -10 z" fill="#ffffff" stroke="currentColor" strokeOpacity="0.4" />
          <motion.text x="182" y="64" textAnchor="middle" className="fill-ink-800 text-[14px] font-semibold">
            {bubbleText}
          </motion.text>
        </g>
      )}
      {definition.caption && (
        <text x="120" y="22" textAnchor="middle" className="fill-current text-[12px] font-medium">
          {definition.caption}
        </text>
      )}
    </svg>
  );
}

function Chair({ x, front }: { x: number; front: boolean }) {
  if (front) {
    return (
      <g stroke="#8a6a4a" strokeWidth="5" strokeLinecap="round" fill="none">
        <path d={`M ${x - 30} 240 V 160 H ${x + 30} V 240 M ${x - 28} 240 V 290 M ${x + 28} 240 V 290`} />
        <path d={`M ${x - 34} 240 h 68`} strokeWidth="7" />
      </g>
    );
  }
  return (
    <g stroke="#8a6a4a" strokeWidth="5" strokeLinecap="round" fill="none">
      <path d={`M ${x - 22} 170 V 240 H ${x + 22} M ${x - 18} 240 V 290 M ${x + 18} 240 V 290`} />
    </g>
  );
}
