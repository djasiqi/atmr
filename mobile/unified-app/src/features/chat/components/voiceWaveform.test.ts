import { describe, expect, it } from "@jest/globals";
import { buildVoiceWaveformHeights, formatVoiceDuration } from "./voiceWaveform";

describe("voiceWaveform", () => {
  it("produit un profil stable pour une même URI", () => {
    const a = buildVoiceWaveformHeights("https://cdn.example/a.m4a", 12);
    const b = buildVoiceWaveformHeights("https://cdn.example/a.m4a", 12);
    expect(a).toEqual(b);
    expect(a).toHaveLength(12);
    expect(a.every((h) => h >= 0.2 && h <= 1)).toBe(true);
  });

  it("formate la durée mm:ss", () => {
    expect(formatVoiceDuration(0)).toBe("0:00");
    expect(formatVoiceDuration(5)).toBe("0:05");
    expect(formatVoiceDuration(75)).toBe("1:15");
  });
});
