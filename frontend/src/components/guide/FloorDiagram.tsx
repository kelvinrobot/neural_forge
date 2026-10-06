import { motion, useReducedMotion } from "framer-motion";
import type { FloorKind } from "./stepAnimations";

function CameraMark() {
  return (
    <g>
      <path d="M 110 112 L 40 20 M 110 112 L 180 20" stroke="currentColor" strokeOpacity="0.25" strokeDasharray="3 4" />
      <rect x="100" y="106" width="20" height="12" rx="3" fill="currentColor" />
      <circle cx="110" cy="112" r="3" fill="#ffffff" />
    </g>
  );
}

export function FloorDiagram({ kind }: { kind: FloorKind }) {
  const reduceMotion = useReducedMotion();
  const loop = { repeat: Infinity, ease: "easeInOut" as const };
  return (
    <figure className="w-full">
      <svg viewBox="0 0 220 126" className="w-full text-ink-500" role="img" aria-label="Floor layout from above">
        <rect x="1" y="1" width="218" height="124" rx="12" fill="none" stroke="currentColor" strokeOpacity="0.2" />
        <CameraMark />
        {kind === "distance" && (
          <g>
            <circle cx="110" cy="26" r="9" fill="#6aa9a6" />
            <path d="M 110 40 V 100" stroke="currentColor" strokeDasharray="4 3" />
            <text x="118" y="74" className="fill-current text-[10px] font-medium">
              ≈ 3 m
            </text>
          </g>
        )}
        {kind === "walk" && (
          <g>
            <path d="M 30 30 H 190" stroke="currentColor" strokeWidth="2" markerEnd="url(#floor-arrow)" markerStart="url(#floor-arrow)" />
            <text x="110" y="48" textAnchor="middle" className="fill-current text-[10px] font-medium">
              4 m path, across and back
            </text>
            <text x="110" y="84" textAnchor="middle" className="fill-current text-[10px]">
              ≈ 3 m from camera
            </text>
            <motion.circle
              cy="30"
              r="7"
              fill="#6aa9a6"
              initial={{ cx: 30 }}
              animate={reduceMotion ? { cx: 110 } : { cx: [30, 190, 30] }}
              transition={{ duration: 6, ...loop }}
            />
          </g>
        )}
        {kind === "tug" && (
          <g>
            <rect x="22" y="22" width="18" height="18" rx="3" fill="#8a6a4a" />
            <text x="31" y="54" textAnchor="middle" className="fill-current text-[9px]">
              chair
            </text>
            <path d="M 44 31 H 176 q 14 0 14 -10 q 0 -10 -14 -6" stroke="currentColor" strokeDasharray="4 3" fill="none" />
            <path d="M 184 24 l 10 10 M 194 24 l -10 10" stroke="#d9480f" strokeWidth="2.5" strokeLinecap="round" />
            <text x="110" y="52" textAnchor="middle" className="fill-current text-[10px] font-medium">
              3 m to the mark, turn, walk back, sit
            </text>
            <motion.circle
              cy="31"
              r="6"
              fill="#6aa9a6"
              initial={{ cx: 40 }}
              animate={reduceMotion ? { cx: 40 } : { cx: [40, 40, 176, 176, 40, 40] }}
              transition={{ duration: 10, times: [0, 0.1, 0.45, 0.55, 0.9, 1], ...loop }}
            />
          </g>
        )}
        <defs>
          <marker id="floor-arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
      </svg>
      <figcaption className="mt-1 text-center text-xs text-ink-500 dark:text-ink-400">View from above: camera at the bottom</figcaption>
    </figure>
  );
}
