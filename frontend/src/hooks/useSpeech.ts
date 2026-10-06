import { useCallback, useEffect, useRef, useState } from "react";
import { useSettingsStore } from "../state/settingsStore";

const supported = typeof window !== "undefined" && "speechSynthesis" in window;

export function pickDefaultVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  return (
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en") && voice.localService) ??
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en")) ??
    voices[0] ??
    null
  );
}

/** Lists available speech voices; they load asynchronously in most browsers. */
export function useVoices() {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>(() => (supported ? window.speechSynthesis.getVoices() : []));
  useEffect(() => {
    if (!supported) return undefined;
    const update = () => setVoices(window.speechSynthesis.getVoices());
    window.speechSynthesis.addEventListener("voiceschanged", update);
    update();
    return () => window.speechSynthesis.removeEventListener("voiceschanged", update);
  }, []);
  return voices;
}

interface SpeakOptions {
  interrupt?: boolean;
}

/** Text-to-speech with captions. Captions are produced even when voice is muted. */
export function useSpeech(onCaption?: (text: string) => void) {
  const voiceEnabled = useSettingsStore((state) => state.voiceEnabled);
  const voiceURI = useSettingsStore((state) => state.voiceURI);
  const voices = useVoices();
  const enabledRef = useRef(voiceEnabled);
  const captionRef = useRef(onCaption);
  enabledRef.current = voiceEnabled;
  captionRef.current = onCaption;

  const voice = voices.find((item) => item.voiceURI === voiceURI) ?? pickDefaultVoice(voices);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(voice);
  voiceRef.current = voice;

  const cancel = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
  }, []);

  const speak = useCallback((text: string, options: SpeakOptions = {}) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    captionRef.current?.(trimmed);
    if (!supported || !enabledRef.current) return;
    if (options.interrupt) window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(trimmed);
    if (voiceRef.current) {
      utterance.voice = voiceRef.current;
      utterance.lang = voiceRef.current.lang;
    }
    utterance.rate = 1;
    window.speechSynthesis.speak(utterance);
  }, []);

  useEffect(() => {
    if (!voiceEnabled) cancel();
  }, [voiceEnabled, cancel]);

  useEffect(() => cancel, [cancel]);

  return { speak, cancel, supported };
}
