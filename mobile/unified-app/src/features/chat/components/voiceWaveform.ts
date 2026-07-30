import { VOICE_WAVEFORM_BAR_COUNT } from "./voiceMessageStyles";

/** Hauteurs relative 0–1, déterministes pour une URI (pas d’analyse audio). */
export function buildVoiceWaveformHeights(
  uri: string,
  barCount = VOICE_WAVEFORM_BAR_COUNT
): number[] {
  let seed = 0;
  for (let i = 0; i < uri.length; i += 1) {
    seed = (seed * 31 + uri.charCodeAt(i)) >>> 0;
  }
  const heights: number[] = [];
  for (let i = 0; i < barCount; i += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const n = (seed % 1000) / 1000;
    // Envelope douce : plus bas aux extrémités, variation au centre.
    const envelope = 0.35 + 0.65 * Math.sin((Math.PI * (i + 0.5)) / barCount);
    heights.push(0.22 + envelope * (0.35 + n * 0.65));
  }
  return heights;
}

export function formatVoiceDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return "0:00";
  const whole = Math.floor(totalSeconds);
  const minutes = Math.floor(whole / 60);
  const seconds = whole % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
