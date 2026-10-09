import { Settings } from "lucide-react";
import { cn } from "@/lib/cn";

type MobileHeaderProps = {
  /** "Ready" when idle, "Listening" while the visual listening state is on. */
  status: string;
  listening: boolean;
};

/**
 * Compact top bar: wordmark + status on the left, settings on the right.
 * The settings control is visual-only for now — there is no settings
 * destination in this phase.
 */
export default function MobileHeader({ status, listening }: MobileHeaderProps) {
  return (
    <header
      className="z-50 shrink-0 border-b border-white/[0.06] bg-surface/80 backdrop-blur-xl"
      style={{ paddingTop: "env(safe-area-inset-top, 0px)" }}
    >
      <div className="mx-auto flex h-14 w-full max-w-[720px] items-center justify-between px-4">
        <div className="flex min-w-0 items-center gap-2">
          <span className="font-headline text-headline-sm uppercase tracking-tight text-on-surface">
            Jaya
          </span>
          <span
            aria-hidden
            className={cn(
              "h-1.5 w-1.5 shrink-0 rounded-full",
              listening
                ? "bg-primary-container motion-safe:animate-pulse"
                : "bg-primary/70",
            )}
          />
          <span
            role="status"
            aria-live="polite"
            className="truncate text-label-md text-on-surface-variant"
          >
            {status}
          </span>
        </div>
        <button
          type="button"
          aria-label="Settings"
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-on-surface-variant transition-colors hover:bg-white/5 active:bg-white/10"
        >
          <Settings aria-hidden className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}
