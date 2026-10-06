import { create } from "zustand";
import type { SkinTone } from "../components/guide/motion";

export type Theme = "light" | "dark";

interface SettingsState {
  theme: Theme;
  voiceEnabled: boolean;
  voiceURI: string | null;
  cameraDeviceId: string | null;
  guideSkinTone: SkinTone;
  toggleTheme: () => void;
  setVoiceEnabled: (enabled: boolean) => void;
  setVoiceURI: (voiceURI: string | null) => void;
  setCameraDeviceId: (deviceId: string | null) => void;
  setGuideSkinTone: (tone: SkinTone) => void;
}

function initialTheme(): Theme {
  if (typeof window === "undefined" || !window.matchMedia) return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export const useSettingsStore = create<SettingsState>((set) => ({
  theme: initialTheme(),
  voiceEnabled: true,
  voiceURI: null,
  cameraDeviceId: null,
  guideSkinTone: "warm",
  toggleTheme: () => set((state) => ({ theme: state.theme === "dark" ? "light" : "dark" })),
  setVoiceEnabled: (voiceEnabled) => set({ voiceEnabled }),
  setVoiceURI: (voiceURI) => set({ voiceURI }),
  setCameraDeviceId: (cameraDeviceId) => set({ cameraDeviceId }),
  setGuideSkinTone: (guideSkinTone) => set({ guideSkinTone }),
}));
