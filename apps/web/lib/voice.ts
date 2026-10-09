"use client";

import { useCallback, useState } from 'react';

/**
 * Presentation states for the voice UI. These describe what the interface
 * shows — they are NOT connected to any speech-recognition backend.
 *
 * - ready:     idle, waiting for the user
 * - listening: visual listening indicator is on (user explicitly started it)
 * - thinking:  reserved for future "processing" indication
 * - speaking:  reserved for future assistant-speech indication
 * - error:     reserved for future microphone/permission failure display
 */
export type VoiceState =
  | "ready"
  | "listening"
  | "thinking"
  | "speaking"
  | "error";

export const VOICE_STATE_LABEL: Record<VoiceState, string> = {
  ready: "Ready",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Unavailable",
};

/**
 * Owns the voice UI state for the current session.
 *
 * Tapping starts the visual listening state; stopping is always explicit
 * (tap again or press the stop control). Nothing here times out on its own,
 * so the UI never implies that a recording has finished.
 *
 * Future speech-to-text integration point: replace `startListening` /
 * `stopListening` internals with real capture calls and drive `state`
 * from the recognizer. The component contract (explicit start/stop,
 * five states) stays the same.
 */
export function useVoiceState() {
  const [state, setState] = useState<VoiceState>("ready");

  const startListening = useCallback(() => {
    setState((current) => (current === "listening" ? current : "listening"));
  }, []);

  const stopListening = useCallback(() => {
    setState((current) => (current === "listening" ? "ready" : current));
  }, []);

  const toggleListening = useCallback(() => {
    setState((current) => (current === "listening" ? "ready" : "listening"));
  }, []);

  return { state, startListening, stopListening, toggleListening };
}
