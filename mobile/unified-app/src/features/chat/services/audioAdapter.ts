import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioPlayer,
  useAudioPlayerStatus,
  useAudioRecorder,
  useAudioRecorderState,
} from "expo-audio";
import {
  safeIsRecording,
  safeRecorderUri,
  safeStopRecorder,
} from "./audioRecorderSafety";

export type ChatAudioFailureReason =
  | "permission_denied"
  | "aborted"
  | "already_recording"
  | "no_active_recording"
  | "no_recording_uri"
  | "recording_error"
  | "playback_error";

export type ChatAudioResult<T> =
  | { ok: true; data: T }
  | { ok: false; reason: ChatAudioFailureReason };

const PLAYBACK_END_TOLERANCE_SECONDS = 0.08;

let activeRecordingOwner: symbol | null = null;
let activePlaybackOwner: symbol | null = null;
let activePlaybackPause: (() => void) | null = null;

async function setRecordingSessionMode(enabled: boolean): Promise<void> {
  await setAudioModeAsync({
    allowsRecording: enabled,
    playsInSilentMode: true,
  });
}

/**
 * Enregistrement vocal (natif uniquement — appelé depuis `ChatComposer.tsx`, non résolu sur le web).
 * Un seul enregistrement global à la fois pour éviter les conflits de session audio.
 */
export function useChatVoiceRecorder() {
  const ownerRef = useRef(Symbol("chat-voice-recorder"));
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder);
  const recorderRef = useRef(recorder);
  recorderRef.current = recorder;

  const abortRecording = useCallback(async (): Promise<void> => {
    const owner = ownerRef.current;
    const current = recorderRef.current;
    const ownsSession = activeRecordingOwner === owner;
    // Ne pas couper la session d'un autre recorder (FAB canal équipe vs ChatComposer).
    // Important : safeIsRecording — un accès direct à isRecording hors try rejette
    // la Promise (onunhandledrejection) si le shared object natif est déjà libéré.
    if (!ownsSession && !safeIsRecording(current)) {
      return;
    }
    try {
      if (ownsSession || safeIsRecording(current)) {
        await safeStopRecorder(current);
      }
    } catch {
      /* ignore */
    } finally {
      if (activeRecordingOwner === owner) {
        activeRecordingOwner = null;
      }
      if (ownsSession) {
        await setRecordingSessionMode(false).catch(() => undefined);
      }
    }
  }, []);

  const startRecording = useCallback(
    async (options?: { isAborted?: () => boolean }): Promise<ChatAudioResult<null>> => {
      const owner = ownerRef.current;
      const current = recorderRef.current;
      if (activeRecordingOwner != null && activeRecordingOwner !== owner) {
        return { ok: false, reason: "already_recording" };
      }
      try {
        const permission = await requestRecordingPermissionsAsync();
        if (!permission.granted) {
          return { ok: false, reason: "permission_denied" };
        }
        if (options?.isAborted?.()) {
          await setRecordingSessionMode(false).catch(() => undefined);
          return { ok: false, reason: "aborted" };
        }
        await setRecordingSessionMode(true);
        if (options?.isAborted?.()) {
          await setRecordingSessionMode(false).catch(() => undefined);
          return { ok: false, reason: "aborted" };
        }
        await current.prepareToRecordAsync();
        if (options?.isAborted?.()) {
          await safeStopRecorder(current);
          await setRecordingSessionMode(false).catch(() => undefined);
          return { ok: false, reason: "aborted" };
        }
        activeRecordingOwner = owner;
        current.record();
        return { ok: true, data: null };
      } catch {
        if (activeRecordingOwner === owner) {
          activeRecordingOwner = null;
        }
        await setRecordingSessionMode(false).catch(() => undefined);
        return { ok: false, reason: "recording_error" };
      }
    },
    []
  );

  const stopRecording = useCallback(async (): Promise<ChatAudioResult<string>> => {
    const owner = ownerRef.current;
    const current = recorderRef.current;
    const ownsSession = activeRecordingOwner === owner;
    if (!ownsSession && !safeIsRecording(current)) {
      return { ok: false, reason: "no_active_recording" };
    }
    try {
      await safeStopRecorder(current);
      const uri = safeRecorderUri(current);
      if (activeRecordingOwner === owner) {
        activeRecordingOwner = null;
      }
      await setRecordingSessionMode(false).catch(() => undefined);
      if (!uri) {
        return { ok: false, reason: "no_recording_uri" };
      }
      return { ok: true, data: uri };
    } catch {
      if (activeRecordingOwner === owner) {
        activeRecordingOwner = null;
      }
      await setRecordingSessionMode(false).catch(() => undefined);
      return { ok: false, reason: "recording_error" };
    }
  }, []);

  // Cleanup au démontage uniquement — éviter d'appeler stop() sur un recorder
  // déjà libéré quand l'identité du hook expo-audio change en cours de vie.
  useEffect(() => {
    return () => {
      void abortRecording();
    };
  }, [abortRecording]);

  return {
    isRecording: recorderState.isRecording,
    durationMillis: recorderState.durationMillis,
    startRecording,
    stopRecording,
    abortRecording,
  };
}

/**
 * Lecture des messages vocaux (natif). Une seule lecture active : les autres sont mises en pause.
 */
export function useChatVoicePlayer(uri: string) {
  const ownerRef = useRef(Symbol("chat-voice-player"));
  const source = useMemo(() => ({ uri }), [uri]);
  const player = useAudioPlayer(source);
  const status = useAudioPlayerStatus(player);

  const releasePlaybackOwnership = useCallback(() => {
    if (activePlaybackOwner === ownerRef.current) {
      activePlaybackOwner = null;
      activePlaybackPause = null;
    }
  }, []);

  const pausePlayback = useCallback(() => {
    try {
      player.pause();
    } catch {
      /* ignore */
    } finally {
      releasePlaybackOwnership();
    }
  }, [player, releasePlaybackOwnership]);

  const playPlayback = useCallback(async (): Promise<ChatAudioResult<null>> => {
    try {
      if (activePlaybackOwner != null && activePlaybackOwner !== ownerRef.current) {
        activePlaybackPause?.();
      }
      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
      const duration = Number(player.duration ?? 0);
      const currentTime = Number(player.currentTime ?? 0);
      const atEnd = duration > 0 && currentTime >= duration - PLAYBACK_END_TOLERANCE_SECONDS;
      if (atEnd) {
        await player.seekTo(0);
      }
      player.play();
      activePlaybackOwner = ownerRef.current;
      activePlaybackPause = () => {
        try {
          player.pause();
        } catch {
          /* ignore */
        }
      };
      return { ok: true, data: null };
    } catch {
      releasePlaybackOwnership();
      return { ok: false, reason: "playback_error" };
    }
  }, [player, releasePlaybackOwnership]);

  const togglePlayback = useCallback(async (): Promise<ChatAudioResult<"playing" | "paused">> => {
    if (player.playing) {
      pausePlayback();
      return { ok: true, data: "paused" };
    }
    const started = await playPlayback();
    if (!started.ok) {
      return started;
    }
    return { ok: true, data: "playing" };
  }, [pausePlayback, playPlayback, player.playing]);

  useEffect(() => {
    if (status.didJustFinish) {
      releasePlaybackOwnership();
    }
  }, [releasePlaybackOwnership, status.didJustFinish]);

  useEffect(() => {
    return () => {
      pausePlayback();
      void player.seekTo(0).catch(() => undefined);
    };
  }, [pausePlayback, player, uri]);

  const durationSeconds = Math.max(0, Number(status.duration ?? player.duration ?? 0));
  const currentTimeSeconds = Math.max(0, Number(status.currentTime ?? player.currentTime ?? 0));
  const progress =
    durationSeconds > 0 ? Math.min(1, Math.max(0, currentTimeSeconds / durationSeconds)) : 0;

  return {
    isPlaying: Boolean(status.playing ?? player.playing),
    durationSeconds,
    currentTimeSeconds,
    progress,
    togglePlayback,
    pausePlayback,
  };
}
