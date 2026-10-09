"use client";

import { useEffect, useState } from "react";

/**
 * Height in CSS pixels currently covered by the on-screen keyboard.
 *
 * Primary mechanism is the `interactiveWidget: "resizes-content"` viewport
 * setting (see app/layout.tsx), which makes the layout viewport itself shrink
 * so fixed content stays visible. This hook is the fallback for browsers that
 * ignore that setting: it tracks `visualViewport` and reports how much of the
 * layout viewport is occluded.
 *
 * Small deltas (URL bar collapse, ~<40px) are treated as zero so the UI does
 * not jump while scrolling.
 */
export function useKeyboardOffset(): number {
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const viewport = window.visualViewport;
    if (!viewport) return;

    const update = () => {
      if (viewport.scale !== 1) return;
      const occluded = window.innerHeight - viewport.height;
      setOffset(occluded > 40 ? Math.round(occluded) : 0);
    };

    update();
    viewport.addEventListener("resize", update);
    viewport.addEventListener("scroll", update);
    return () => {
      viewport.removeEventListener("resize", update);
      viewport.removeEventListener("scroll", update);
    };
  }, []);

  return offset;
}
