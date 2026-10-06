import { motion } from "framer-motion";
import type { Step } from "../../api/types";
import { Icon } from "../ui/Icon";
import { BodyGuide } from "./BodyGuide";
import { FaceGuide } from "./FaceGuide";
import { FloorDiagram } from "./FloorDiagram";
import { HandGuide } from "./HandGuide";
import { SKIN_TONES, type SkinTone } from "./motion";
import { SkinGuide } from "./SkinGuide";
import { guideForStep, type GuideSide, type GuideSpec } from "./stepAnimations";

interface GuideFigureProps {
  examId: string;
  step: Step;
  playing: boolean;
  skinTone?: SkinTone;
}

function SideLabel({ side }: { side: GuideSide }) {
  // The figure is a mirror image, so the patient's right appears on the screen's left.
  const onScreenLeft = side === "right";
  return (
    <div className={`absolute top-2 flex items-center gap-1 rounded-full bg-brand-600 px-2.5 py-1 text-xs font-semibold text-white shadow ${onScreenLeft ? "left-2" : "right-2 flex-row-reverse"}`}>
      <span aria-hidden="true">{onScreenLeft ? "←" : "→"}</span>
      {side === "right" ? "Right" : "Left"}
    </div>
  );
}

function Figure({ spec, playing, skin }: { spec: GuideSpec; playing: boolean; skin: string }) {
  switch (spec.figure) {
    case "face":
      return <FaceGuide pose={spec.pose} playing={playing} skinTone={skin} />;
    case "hand":
      return <HandGuide pose={spec.pose} side={spec.side} playing={playing} skinTone={skin} />;
    case "body":
      return <BodyGuide pose={spec.pose} side={spec.side} playing={playing} skinTone={skin} />;
    case "skin":
      return <SkinGuide pose={spec.pose} side={spec.side} playing={playing} skinTone={skin} />;
  }
}

export function GuideFigure({ examId, step, playing, skinTone = "warm" }: GuideFigureProps) {
  const spec = guideForStep(examId, step.id, step.view);
  const skin = SKIN_TONES[skinTone];
  return (
    <div className="flex flex-col gap-3">
      <div className="relative aspect-square w-full overflow-hidden rounded-2xl bg-gradient-to-b from-brand-50 to-white dark:from-ink-800 dark:to-ink-900">
        <Figure key={`${spec.figure}-${spec.pose}-${spec.side ?? ""}`} spec={spec} playing={playing} skin={skin} />
        {spec.side && <SideLabel side={spec.side} />}
        <div className="absolute bottom-2 left-2 flex gap-1.5">
          {spec.hold && (
            <motion.span
              className="rounded-full bg-white/90 px-2.5 py-1 text-xs font-semibold text-brand-700 shadow dark:bg-ink-900/90 dark:text-brand-300"
              animate={{ opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 2.4, repeat: Infinity }}
            >
              Hold
            </motion.span>
          )}
          {spec.badge === "timer" && (
            <span className="inline-flex items-center gap-1 rounded-full bg-white/90 px-2.5 py-1 text-xs font-semibold text-ink-700 shadow dark:bg-ink-900/90 dark:text-ink-200">
              <Icon name="timer" size={14} /> Keep holding
            </span>
          )}
          {spec.badge === "light" && (
            <span className="inline-flex items-center gap-1 rounded-full bg-white/90 px-2.5 py-1 text-xs font-semibold text-ink-700 shadow dark:bg-ink-900/90 dark:text-ink-200">
              <Icon name="light" size={14} /> Light into the mouth
            </span>
          )}
        </div>
      </div>
      {spec.figure === "body" && spec.floor && <FloorDiagram kind={spec.floor} />}
    </div>
  );
}
