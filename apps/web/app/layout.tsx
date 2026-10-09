import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  display: "swap",
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "JAYA",
  description: "JAYA — your personal AI, ready when you are.",
  applicationName: "JAYA",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0f131c",
  colorScheme: "dark",
  // Shrink the layout viewport when the on-screen keyboard opens so the
  // fixed composer stays visible instead of being covered.
  interactiveWidget: "resizes-content",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${inter.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-surface font-body text-on-surface">
        {children}
      </body>
    </html>
  );
}
