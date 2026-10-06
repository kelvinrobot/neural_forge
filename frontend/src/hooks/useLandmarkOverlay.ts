import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState, type RefObject } from "react";
import type { StepView } from "../api/types";

const MEDIAPIPE_VERSION = "0.10.14";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URLS = {
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
};
const MIN_FRAME_GAP_MS = 33;

export type LandmarkKind = "face" | "hand" | "pose";
export type OverlayHighlight = "eyes" | "mouth" | "face" | null;
export type OverlayStatus = "loading" | "ready" | "unavailable";

export function landmarkKindsForExam(examId: string): LandmarkKind[] {
  if (examId === "hands") return ["hand"];
  if (examId === "movement") return ["pose", "hand"];
  if (examId === "posture") return ["pose"];
  return ["face"];
}

function kindsForView(view: StepView | null): LandmarkKind[] {
  switch (view) {
    case "face":
    case "face_close":
    case "mouth":
    case "skin":
      return ["face"];
    case "hands":
    case "hand_close":
      return ["hand"];
    case "upper_body":
      return ["pose", "hand"];
    case "full_body":
      return ["pose"];
    default:
      return [];
  }
}

interface Landmarkers {
  face?: FaceLandmarker;
  hand?: HandLandmarker;
  pose?: PoseLandmarker;
}

async function withDelegateFallback<T>(create: (delegate: "GPU" | "CPU") => Promise<T>): Promise<T> {
  try {
    return await create("GPU");
  } catch {
    return create("CPU");
  }
}

async function loadLandmarkers(kinds: LandmarkKind[]): Promise<Landmarkers> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
  const result: Landmarkers = {};
  if (kinds.includes("face")) {
    result.face = await withDelegateFallback((delegate) =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URLS.face, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
      }),
    );
  }
  if (kinds.includes("hand")) {
    result.hand = await withDelegateFallback((delegate) =>
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URLS.hand, delegate },
        runningMode: "VIDEO",
        numHands: 2,
      }),
    );
  }
  if (kinds.includes("pose")) {
    result.pose = await withDelegateFallback((delegate) =>
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URLS.pose, delegate },
        runningMode: "VIDEO",
        numPoses: 1,
      }),
    );
  }
  return result;
}

const MESH = "rgba(160, 239, 231, 0.28)";
const CONTOUR = "rgba(160, 239, 231, 0.75)";
const ACCENT = "#2fc4be";

function drawFace(drawer: DrawingUtils, landmarks: NormalizedLandmark[], scale: number, highlight: OverlayHighlight) {
  drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: MESH, lineWidth: 0.5 * scale });
  drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: CONTOUR, lineWidth: 1.2 * scale });
  const eyes = [
    FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
  ];
  const strongEyes = highlight === "eyes";
  eyes.forEach((connections) =>
    drawer.drawConnectors(landmarks, connections, { color: strongEyes ? ACCENT : CONTOUR, lineWidth: (strongEyes ? 2.4 : 1) * scale }),
  );
  const strongMouth = highlight === "mouth";
  drawer.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
    color: strongMouth ? ACCENT : CONTOUR,
    lineWidth: (strongMouth ? 2.4 : 1) * scale,
  });
  if (strongEyes || strongMouth) {
    const indices = strongEyes ? [468, 473, 33, 133, 263, 362, 159, 145, 386, 374] : [61, 291, 0, 17, 13, 14];
    drawer.drawLandmarks(
      indices.map((index) => landmarks[index]).filter(Boolean),
      { color: "#ffffff", fillColor: ACCENT, radius: 2.4 * scale, lineWidth: 1 * scale },
    );
  }
}

interface OverlayOptions {
  examId: string;
  view: StepView | null;
  highlight: OverlayHighlight;
  enabled: boolean;
}

/** Draws in-browser landmark overlays for visual feedback only; nothing is sent to the backend. */
export function useLandmarkOverlay(
  videoRef: RefObject<HTMLVideoElement>,
  canvasRef: RefObject<HTMLCanvasElement>,
  { examId, view, highlight, enabled }: OverlayOptions,
): OverlayStatus {
  const [status, setStatus] = useState<OverlayStatus>("loading");
  const landmarkersRef = useRef<Landmarkers>({});
  const settingsRef = useRef({ view, highlight });
  settingsRef.current = { view, highlight };

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    setStatus("loading");
    loadLandmarkers(landmarkKindsForExam(examId))
      .then((landmarkers) => {
        if (cancelled) {
          Object.values(landmarkers).forEach((landmarker) => landmarker?.close());
          return;
        }
        landmarkersRef.current = landmarkers;
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("unavailable");
      });
    return () => {
      cancelled = true;
      Object.values(landmarkersRef.current).forEach((landmarker) => landmarker?.close());
      landmarkersRef.current = {};
    };
  }, [examId, enabled]);

  useEffect(() => {
    if (status !== "ready") return undefined;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!video || !canvas || !context) return undefined;
    const drawer = new DrawingUtils(context);
    let stopped = false;
    let lastRun = 0;
    let frameHandle = 0;
    const useVideoCallback = "requestVideoFrameCallback" in video;

    const schedule = () => {
      if (stopped) return;
      frameHandle = useVideoCallback ? video.requestVideoFrameCallback(run) : window.requestAnimationFrame(run);
    };

    const run = () => {
      const now = performance.now();
      if (now - lastRun >= MIN_FRAME_GAP_MS && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth) {
        lastRun = now;
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        context.clearRect(0, 0, canvas.width, canvas.height);
        const scale = canvas.width / 640;
        const { view: currentView, highlight: currentHighlight } = settingsRef.current;
        const kinds = kindsForView(currentView);
        const { face, hand, pose } = landmarkersRef.current;
        if (kinds.includes("face") && face) {
          face.detectForVideo(video, now).faceLandmarks.forEach((landmarks) => drawFace(drawer, landmarks, scale, currentHighlight));
        }
        if (kinds.includes("pose") && pose) {
          pose.detectForVideo(video, now).landmarks.forEach((landmarks) => {
            drawer.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: CONTOUR, lineWidth: 2.5 * scale });
            drawer.drawLandmarks(landmarks, { color: "#ffffff", fillColor: ACCENT, radius: 3 * scale, lineWidth: 1 * scale });
          });
        }
        if (kinds.includes("hand") && hand) {
          hand.detectForVideo(video, now).landmarks.forEach((landmarks) => {
            drawer.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: CONTOUR, lineWidth: 2 * scale });
            drawer.drawLandmarks(landmarks, { color: "#ffffff", fillColor: ACCENT, radius: 2.5 * scale, lineWidth: 1 * scale });
          });
        }
      }
      schedule();
    };

    schedule();
    return () => {
      stopped = true;
      if (useVideoCallback) video.cancelVideoFrameCallback(frameHandle);
      else window.cancelAnimationFrame(frameHandle);
      context.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [status, videoRef, canvasRef]);

  return enabled ? status : "unavailable";
}
