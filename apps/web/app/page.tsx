"use client";

import { useEffect, useRef, useState } from "react";
import { MotionConfig, motion } from "framer-motion";
import MobileHeader from "@/components/MobileHeader";
import LightfallBackground from "@/components/LightfallBackground";
import ConversationPanel, {
  type ChatMessage,
} from "@/components/ConversationPanel";
import PromptBar, { type PromptBarSource } from "@/components/PromptBar";
import { useVoiceState } from "@/lib/voice";
import { useKeyboardOffset } from "@/lib/use-keyboard-offset";
import {
  Calendar03Icon,
  File02Icon,
  Globe02Icon,
} from "@hugeicons/core-free-icons";

/**
 * Sources offered by the composer's plus menu. No attach row: JAYA has no
 * file-upload feature in this phase, so nothing here fabricates one.
 */
const PROMPT_SOURCES: PromptBarSource[] = [
  { key: "web", name: "Web search", description: "Live results", icon: Globe02Icon },
  { key: "docs", name: "Documents", description: "Specs, notes, briefs", icon: File02Icon },
  { key: "calendar", name: "Calendar", description: "Events and availability", icon: Calendar03Icon },
];

const PROMPT_MODELS = [{ key: "jaya", name: "JAYA", tag: "Default" }];

/**
 * Restrained welcome state. Shown only when no messages exist yet —
 * compact, centered in the available conversation space.
 */
function WelcomeState() {
  return (
    <div className="relative flex flex-1 flex-col items-center justify-center px-6 py-12 text-center">
      {/* Faint atmospheric bloom behind the welcome text only. */}
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-1/2 h-56 w-56 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/[0.07] blur-3xl"
      />
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="relative"
      >
        <h1 className="font-headline text-headline-md tracking-tight text-on-surface">
          What can I help with?
        </h1>
        <p className="mt-2 text-body-md text-on-surface-variant">
          Ask, explore, or think something through.
        </p>
      </motion.div>
    </div>
  );
}

export default function Home() {
  const { state: voiceState, toggleListening, stopListening } = useVoiceState();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const keyboardOffset = useKeyboardOffset();

  const endRef = useRef<HTMLDivElement | null>(null);
  const idRef = useRef(0);

  const listening = voiceState === "listening";

  const sendMessage = (text: string) => {
    idRef.current += 1;
    setMessages((current) => [
      ...current,
      { id: `${Date.now()}-${idRef.current}`, text },
    ]);
  };

  // Keep the latest message reachable: scroll it into view above the
  // composer whenever messages arrive or the keyboard opens/closes.
  useEffect(() => {
    if (messages.length === 0 && keyboardOffset === 0) return;
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    endRef.current?.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "end",
    });
  }, [messages, keyboardOffset]);

  return (
    <MotionConfig reducedMotion="never">
      <div className="flex h-dvh flex-col overflow-x-hidden text-on-surface">
        <LightfallBackground />
        <MobileHeader
          status={listening ? "Listening" : "Ready"}
          listening={listening}
        />

        <main className="relative z-10 flex min-h-0 flex-1 flex-col overflow-y-auto">
          <div className="mx-auto flex min-h-full w-full max-w-[720px] flex-1 flex-col px-4 pb-44">
            {messages.length === 0 ? (
              <WelcomeState />
            ) : (
              <>
                <ConversationPanel messages={messages} />
                {/* Scroll anchor: the newest content always ends here. */}
                <div ref={endRef} aria-hidden className="h-px w-full" />
              </>
            )}
          </div>
        </main>

        {/* Subtle listening indicator above the composer (visual only). */}
        {listening && (
          <div
            className="pointer-events-none fixed inset-x-0 z-20"
            style={{
              bottom: `calc(104px + max(env(safe-area-inset-bottom, 0px), 0.75rem) + ${keyboardOffset}px)`,
            }}
          >
            <div className="mx-auto flex w-full max-w-[720px] justify-center px-4">
              <p className="pointer-events-auto flex items-center gap-3 rounded-full border border-white/[0.06] bg-surface-container px-4 py-1.5 text-label-md text-on-surface-variant">
                <span
                  aria-hidden
                  className="h-1.5 w-1.5 rounded-full bg-primary-container motion-safe:animate-pulse"
                />
                <span role="status" aria-live="polite">
                  Listening — visual indicator only
                </span>
                <button
                  type="button"
                  onClick={stopListening}
                  className="rounded-full px-2 py-1 text-label-md text-primary transition-colors active:text-on-surface"
                >
                  Stop
                </button>
              </p>
            </div>
          </div>
        )}

        <div
          className="pointer-events-none fixed inset-x-0 bottom-0 z-20"
          style={{
            paddingBottom: `calc(max(env(safe-area-inset-bottom, 0px), 0.75rem) + ${keyboardOffset}px)`,
          }}
        >
          <div className="mx-auto w-full max-w-[720px] px-4">
            <PromptBar
              placeholder="Ask JAYA anything..."
              sources={PROMPT_SOURCES}
              models={PROMPT_MODELS}
              onSend={(text) => sendMessage(text)}
              onStop={stopListening}
              onDictate={() => {
                // Visual-only voice state: toggles the JAYA listening
                // indicator. No recording or transcription is connected.
                toggleListening();
              }}
              background="#1c2028"
              color="#dfe2ee"
              menuBackground="#262a33"
              sparkColor="#8ed5ff"
              width={720}
              radius={24}
              maxRows={5}
              className="pointer-events-auto"
            />
          </div>
        </div>
      </div>
    </MotionConfig>
  );
}
