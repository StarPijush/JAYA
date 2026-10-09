"use client";

import dynamic from "next/dynamic";
import { memo, useEffect, useState, useSyncExternalStore } from "react";

// WebGL canvas is client-only: the loading fallback renders on the server and
// the first client render, keeping hydration deterministic.
const Lightfall = dynamic(() => import("./Lightfall"), {
  ssr: false,
  loading: () => (
    <div
      aria-hidden
      className="h-full w-full"
      style={{ backgroundColor: "#05070d" }}
    />
  ),
});

/**
 * Exact starting props from the integration spec. Module scope keeps the
 * array reference stable so the WebGL effect never re-runs on parent
 * re-renders (e.g. every composer keystroke).
 */
const LIGHTFALL_COLORS = ["#A6C8FF", "#5227FF", "#FF9FFC"];

/** True only after hydration, so the first render matches the server. */
function useMounted(): boolean {
  return useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
}

/** Cached WebGL2 support probe. */
let cachedCapable: boolean | null = null;
function isCapable(): boolean {
  if (cachedCapable !== null) return cachedCapable;
  if (typeof window === "undefined" || typeof document === "undefined") {
    cachedCapable = false;
    return cachedCapable;
  }
  const probe = document.createElement("canvas");
  cachedCapable = !!probe.getContext("webgl2");
  return cachedCapable;
}

/**
 * Full-viewport Lightfall layer for the JAYA background.
 *
 * Memoized with zero props so parent re-renders (composer input, messages)
 * can never recreate the single WebGL renderer. Stays click-through at all
 * times; cursor glow is tracked at window level inside Lightfall itself.
 * Falls back to a static dark background only when WebGL2 is unavailable.
 */
function LightfallBackground() {
  // Server and first client render show the static fallback; the canvas only
  // mounts after hydration when animation is appropriate — no mismatch.
  const mounted = useMounted();
  // Pauses frame rendering while the tab is hidden (read via ref in
  // Lightfall, so this never rebuilds the renderer).
  const [pageHidden, setPageHidden] = useState(false);

  useEffect(() => {
    const onVisibility = () => setPageHidden(document.hidden);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  const ready = mounted && isCapable();

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 z-0 bg-[#05070d]"
    >
      {ready ? (
        <Lightfall
          colors={LIGHTFALL_COLORS}
          backgroundColor="#0A29FF"
          speed={0.5}
          streakCount={2}
          streakWidth={1}
          streakLength={1}
          glow={1}
          density={0.6}
          twinkle={1}
          zoom={3}
          backgroundGlow={0.5}
          opacity={1}
          mouseInteraction
          mouseStrength={0.5}
          mouseRadius={1}
          mouseDampening={0.15}
          dpr={1.5}
          paused={pageHidden}
        />
      ) : (
        <div
          className="h-full w-full"
          style={{ backgroundColor: "#05070d" }}
        />
      )}
      {/* Subtle dark overlay so text and controls keep full contrast. */}
      <div className="absolute inset-0 bg-[#04060c]/60" />
    </div>
  );
}

export default memo(LightfallBackground);
