# JAYA Mobile Web UI

Mobile-only web interface for JAYA. Next.js 16 + React 19 + TypeScript +
Tailwind CSS v4 + Framer Motion + Lucide.

This is a **UI-only phase**: nothing here talks to a backend, an LLM, or any
speech-recognition engine. Messages stay in local component state.

## Run it

```powershell
cd apps\web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), ideally in a mobile
viewport (320–430px wide). On wider screens the UI constrains itself to a
centered 430px column.

## Verify it

```powershell
npm run build   # production build (runs TypeScript + lint internally)
npx tsc --noEmit
npx eslint .
```

## What works

- Sending a message (appears locally, input clears, empty input can't send)
- Quick-action pills fill the composer
- Voice orb toggles a visual ready/listening state; an explicit Stop control
  ends it — nothing times out on its own
- Composer stays visible when the Android keyboard opens
  (`interactive-widget=resizes-content` plus a `visualViewport` fallback)
- Reduced-motion preferences disable ambient animation

## Intentionally unconnected

Backend API, LLM, speech-to-text, text-to-speech, wake word, database, auth,
attachments (the `+` icon is decorative). The `useVoiceState` hook in
`lib/voice.ts` is the documented integration point for future voice work.
