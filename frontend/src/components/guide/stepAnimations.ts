import type { StepView } from "../../api/types";

export type GuideSide = "right" | "left";

export type FacePoseName =
  | "neutral"
  | "smile"
  | "eyebrows"
  | "eyes_closed"
  | "cheeks"
  | "expressions"
  | "talking"
  | "straight"
  | "gaze_left"
  | "gaze_right"
  | "gaze_up"
  | "gaze_down"
  | "upgaze"
  | "close_up"
  | "lower_lid"
  | "mouth_closed"
  | "open_wide"
  | "tongue_out"
  | "aaah"
  | "gums";

export type HandPoseName =
  | "palms"
  | "backs"
  | "side_view"
  | "tap"
  | "flip"
  | "nail_baseline"
  | "nail_press"
  | "refill"
  | "window"
  | "lap";

export type BodyPoseName =
  | "front"
  | "side"
  | "back"
  | "arms_side"
  | "knee_bends"
  | "balance_open"
  | "balance_closed"
  | "single_leg"
  | "walking"
  | "tug"
  | "sit_to_stand"
  | "rest"
  | "postural"
  | "drift"
  | "nose";

export type SkinPoseName = "turn_head" | "forearms_inner" | "forearms_outer" | "area";

export type FloorKind = "distance" | "walk" | "tug";

export type GuideBadge = "timer" | "light";

interface SpecBase {
  side?: GuideSide;
  hold?: boolean;
  badge?: GuideBadge;
}

export type GuideSpec =
  | (SpecBase & { figure: "face"; pose: FacePoseName })
  | (SpecBase & { figure: "hand"; pose: HandPoseName })
  | (SpecBase & { figure: "body"; pose: BodyPoseName; floor?: FloorKind })
  | (SpecBase & { figure: "skin"; pose: SkinPoseName });

const face = (pose: FacePoseName, extra: SpecBase = {}): GuideSpec => ({ figure: "face", pose, ...extra });
const hand = (pose: HandPoseName, extra: SpecBase = {}): GuideSpec => ({ figure: "hand", pose, ...extra });
const body = (pose: BodyPoseName, extra: SpecBase & { floor?: FloorKind } = {}): GuideSpec => ({ figure: "body", pose, ...extra });
const skin = (pose: SkinPoseName, extra: SpecBase = {}): GuideSpec => ({ figure: "skin", pose, ...extra });

const STEP_GUIDES: Record<string, Record<string, GuideSpec>> = {
  face: {
    neutral: face("neutral", { hold: true }),
    smile: face("smile", { hold: true }),
    eyebrows: face("eyebrows", { hold: true }),
    eyes_closed: face("eyes_closed", { hold: true }),
    cheeks: face("cheeks", { hold: true }),
    expressions: face("expressions"),
    natural: face("talking"),
  },
  eyes: {
    straight: face("straight"),
    gaze_left: face("gaze_left", { side: "left", hold: true }),
    gaze_right: face("gaze_right", { side: "right", hold: true }),
    gaze_up: face("gaze_up", { hold: true }),
    gaze_down: face("gaze_down", { hold: true }),
    upgaze: face("upgaze", { hold: true, badge: "timer" }),
    close_up: face("close_up"),
    lower_lid: face("lower_lid", { hold: true }),
  },
  mouth: {
    closed: face("mouth_closed", { hold: true }),
    open_wide: face("open_wide", { hold: true }),
    tongue_out: face("tongue_out", { hold: true }),
    inside: face("aaah", { badge: "light" }),
    gums: face("gums", { hold: true }),
  },
  hands: {
    palms: hand("palms", { hold: true }),
    backs: hand("backs", { hold: true }),
    side_right: hand("side_view", { side: "right", hold: true }),
    side_left: hand("side_view", { side: "left", hold: true }),
    window: hand("window", { hold: true }),
    nail_baseline: hand("nail_baseline", { hold: true }),
    nail_press: hand("nail_press"),
    refill: hand("refill"),
  },
  movement: {
    rest: body("rest"),
    postural: body("postural", { hold: true }),
    drift: body("drift", { hold: true }),
    tap_right: hand("tap", { side: "right" }),
    tap_left: hand("tap", { side: "left" }),
    flip_right: hand("flip", { side: "right" }),
    flip_left: hand("flip", { side: "left" }),
    nose_right: body("nose", { side: "right" }),
    nose_left: body("nose", { side: "left" }),
    quiet: hand("lap", { hold: true }),
  },
  posture: {
    front: body("front", { hold: true, floor: "distance" }),
    side: body("side", { hold: true, floor: "distance" }),
    back: body("back", { hold: true, floor: "distance" }),
    arms_side: body("arms_side", { floor: "distance" }),
    knee_bends: body("knee_bends", { floor: "distance" }),
    balance_open: body("balance_open", { hold: true, floor: "distance" }),
    balance_closed: body("balance_closed", { hold: true, floor: "distance" }),
    single_right: body("single_leg", { side: "right", hold: true, floor: "distance" }),
    single_left: body("single_leg", { side: "left", hold: true, floor: "distance" }),
    walking: body("walking", { floor: "walk" }),
    tug: body("tug", { floor: "tug" }),
    sit_to_stand: body("sit_to_stand", { floor: "distance" }),
  },
  skin: {
    face_front: face("neutral", { hold: true }),
    face_left: skin("turn_head", { side: "left", hold: true }),
    face_right: skin("turn_head", { side: "right", hold: true }),
    forearms_inner: skin("forearms_inner", { hold: true }),
    forearms_outer: skin("forearms_outer", { hold: true }),
    area: skin("area", { hold: true }),
  },
};

const VIEW_FALLBACKS: Record<StepView, GuideSpec> = {
  face: face("neutral", { hold: true }),
  face_close: face("close_up"),
  mouth: face("open_wide", { hold: true }),
  hands: hand("palms", { hold: true }),
  hand_close: hand("nail_baseline", { hold: true }),
  upper_body: body("rest"),
  full_body: body("front", { hold: true, floor: "distance" }),
  skin: skin("area", { hold: true }),
};

export function guideForStep(examId: string, stepId: string, view: StepView): GuideSpec {
  return STEP_GUIDES[examId]?.[stepId] ?? VIEW_FALLBACKS[view];
}
