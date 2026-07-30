/**
 * Garde-fous autour des shared objects expo-audio.
 * Sur Android, un AudioRecorder libéré rejette encore via le bridge
 * (« Cannot use shared object that was already released » / cast Integer).
 */

export type RecorderLike = {
  isRecording: boolean;
  uri: string | null;
  stop: () => Promise<void> | void;
  prepareToRecordAsync?: () => Promise<void>;
  record?: () => void;
};

export function isReleasedRecorderError(error: unknown): boolean {
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : error && typeof error === "object" && "message" in error
          ? String((error as { message: unknown }).message)
          : "";
  const normalized = message.toLowerCase();
  return (
    normalized.includes("already released") ||
    normalized.includes("cannot be cast to type expo.modules.audio.audiorecorder") ||
    normalized.includes("shared object")
  );
}

export function safeIsRecording(recorder: RecorderLike): boolean {
  try {
    return Boolean(recorder.isRecording);
  } catch (error) {
    if (isReleasedRecorderError(error)) return false;
    return false;
  }
}

export function safeRecorderUri(recorder: RecorderLike): string | null {
  try {
    const rawUri = recorder.uri;
    return typeof rawUri === "string" && rawUri.trim().length > 0 ? rawUri : null;
  } catch {
    return null;
  }
}

export async function safeStopRecorder(recorder: RecorderLike): Promise<void> {
  try {
    await recorder.stop();
  } catch (error) {
    if (isReleasedRecorderError(error)) return;
    /* stop idempotent / session déjà arrêtée */
  }
}
