import type { QualityCode } from "../api/types";
import type { IconName } from "../components/ui/Icon";

export interface QualityHint {
  title: string;
  hint: string;
  icon: IconName;
}

export const qualityHints: Record<QualityCode, QualityHint> = {
  too_dark: { title: "More light needed", hint: "Turn on a light or face a window.", icon: "brightness" },
  too_bright: { title: "Too bright", hint: "Move away from direct sunlight or strong lamps.", icon: "sun" },
  blurry: { title: "Image is blurry", hint: "Hold still and keep the camera steady.", icon: "blur" },
  no_face: { title: "Face not visible", hint: "Face the camera so your whole face is in view.", icon: "face" },
  face_too_far: { title: "Move closer", hint: "Bring your face closer to the camera.", icon: "arrowIn" },
  face_too_close: { title: "Move back", hint: "Move a little further from the camera.", icon: "arrowOut" },
  need_closer_eyes: { title: "Closer to your eyes", hint: "Bring the camera closer until your eyes fill the frame.", icon: "eye" },
  need_closer_mouth: { title: "Closer to your mouth", hint: "Bring the camera closer to your mouth.", icon: "mouth" },
  head_turned: { title: "Face the camera", hint: "Turn your head to look straight at the camera.", icon: "rotate" },
  head_tilted: { title: "Level your head", hint: "Keep your head upright, not tilted to one side.", icon: "level" },
  hands_missing_one: { title: "Show both hands", hint: "One hand is out of view. Show both hands.", icon: "hands" },
  hands_missing_two: { title: "Hands not visible", hint: "Hold both hands up in front of the camera.", icon: "hands" },
  hand_too_far: { title: "Hand closer", hint: "Bring your hand closer to the camera.", icon: "arrowIn" },
  upper_body_missing: { title: "Show your upper body", hint: "Move back so your head, shoulders and hands are visible.", icon: "body" },
  full_body_missing: { title: "Step into view", hint: "Stand where the camera can see your whole body.", icon: "body" },
  full_body_too_small: { title: "Come closer", hint: "Walk a little closer to the camera.", icon: "arrowIn" },
  full_body_cut: { title: "Step back", hint: "Step back so your head and feet are both in view.", icon: "arrowOut" },
  no_frames: { title: "No camera images", hint: "Check that the camera is connected and not used by another app.", icon: "camera" },
};

export function qualityHintFor(code: string): QualityHint {
  return (qualityHints as Record<string, QualityHint>)[code] ?? {
    title: "Adjust your position",
    hint: "Follow the outline on the screen.",
    icon: "camera",
  };
}
