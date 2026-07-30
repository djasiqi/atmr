import { describe, expect, it, jest } from "@jest/globals";
import {
  isReleasedRecorderError,
  safeIsRecording,
  safeRecorderUri,
  safeStopRecorder,
  type RecorderLike,
} from "./audioRecorderSafety";

describe("audioRecorderSafety", () => {
  it("détecte les erreurs shared object déjà libéré", () => {
    expect(
      isReleasedRecorderError(
        new Error(
          "The 1st argument cannot be cast to type expo.modules.audio.AudioRecorder (received class java.lang.Integer) → Caused by: Cannot use shared object that was already released"
        )
      )
    ).toBe(true);
    expect(isReleasedRecorderError(new Error("permission denied"))).toBe(false);
  });

  it("safeIsRecording ne laisse pas fuiter une exception released", () => {
    const recorder: RecorderLike = {
      get isRecording() {
        throw new Error("Cannot use shared object that was already released");
      },
      uri: null,
      stop: async () => undefined,
    };
    expect(safeIsRecording(recorder)).toBe(false);
  });

  it("safeRecorderUri tolère un getter uri cassé", () => {
    const recorder: RecorderLike = {
      isRecording: false,
      get uri() {
        throw new Error("Cannot use shared object that was already released");
      },
      stop: async () => undefined,
    };
    expect(safeRecorderUri(recorder)).toBeNull();
  });

  it("safeStopRecorder avale le rejet already released", async () => {
    const stop = jest.fn(async () => {
      throw new Error("Cannot use shared object that was already released");
    });
    const recorder: RecorderLike = {
      isRecording: true,
      uri: null,
      stop,
    };
    await expect(safeStopRecorder(recorder)).resolves.toBeUndefined();
    expect(stop).toHaveBeenCalled();
  });
});
