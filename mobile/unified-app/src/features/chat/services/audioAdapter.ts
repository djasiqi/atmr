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

async function setPlaybackSessionMode(): Promise<void> {
  // Quitter le mode micro (sinon Android/iOS refusent souvent la lecture).
  await setAudioModeAsync({
    allowsRecording: false,
    playsInSilentMode: true,
  });
}

function isRemoteAudioUri(uri: string): boolean {
  return /^https?:\/\//i.test(uri.trim());
}

async function waitUntil(
  predicate: () => boolean,
  timeoutMs: number,
  stepMs = 60
): Promise<boolean> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (predicate()) return true;
    await new Promise((resolve) => setTimeout(resolve, stepMs));
  }
  return predicate();
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
  const downloadFirst = isRemoteAudioUri(uri);
  const player = useAudioPlayer(source, {
    downloadFirst,
    updateInterval: 200,
  });
  const status = useAudioPlayerStatus(player);
  const playerRef = useRef(player);
  playerRef.current = player;
  const statusRef = useRef(status);
  statusRef.current = status;

  const releasePlaybackOwnership = useCallback(() => {
    if (activePlaybackOwner === ownerRef.current) {
      activePlaybackOwner = null;
      activePlaybackPause = null;
    }
  }, []);

  const pausePlayback = useCallback(() => {
    try {
      playerRef.current.pause();
    } catch {
      /* ignore */
    } finally {
      releasePlaybackOwnership();
    }
  }, [releasePlaybackOwnership]);

  const playPlayback = useCallback(async (): Promise<ChatAudioResult<null>> => {
    const current = playerRef.current;
    try {
      if (activePlaybackOwner != null && activePlaybackOwner !== ownerRef.current) {
        activePlaybackPause?.();
      }
      await setPlaybackSessionMode();

      const ready = await waitUntil(() => {
        const s = statusRef.current;
        const loaded = Boolean(
          s.isLoaded ||
            (typeof (current as { isLoaded?: boolean }).isLoaded === "boolean" &&
              (current as { isLoaded?: boolean }).isLoaded)
        );
        // Durée connue = source exploitable même si isLoaded tarde.
        const hasDuration = Number(s.duration ?? current.duration ?? 0) > 0;
        return loaded || hasDuration || !downloadFirst;
      }, downloadFirst ? 6000 : 1500);

      if (!ready && downloadFirst) {
        // Dernière tentative : forcer un replace n’est pas exposé ; on joue quand même.
      }

      const duration = Number(statusRef.current.duration ?? current.duration ?? 0);
      const currentTime = Number(statusRef.current.currentTime ?? current.currentTime ?? 0);
      const atEnd = duration > 0 && currentTime >= duration - PLAYBACK_END_TOLERANCE_SECONDS;
      if (atEnd || currentTime > 0) {
        try {
          await current.seekTo(0);
        } catch {
          /* ignore */
        }
      }

      current.play();
      // Android : premier play parfois no-op — second essai court.
      await new Promise((resolve) => setTimeout(resolve, 120));
      const playingNow = Boolean(
        statusRef.current.playing || current.playing
      );
      if (!playingNow) {
        current.play();
      }

      activePlaybackOwner = ownerRef.current;
      activePlaybackPause = () => {
        try {
          playerRef.current.pause();
        } catch {
          /* ignore */
        }
      };
      return { ok: true, data: null };
    } catch {
      releasePlaybackOwnership();
      return { ok: false, reason: "playback_error" };
    }
  }, [downloadFirst, releasePlaybackOwnership]);

  const togglePlayback = useCallback(async (): Promise<ChatAudioResult<"playing" | "paused">> => {
    if (statusRef.current.playing || playerRef.current.playing) {
      pausePlayback();
      return { ok: true, data: "paused" };
    }
    const started = await playPlayback();
    if (!started.ok) {
      return started;
    }
    return { ok: true, data: "playing" };
  }, [pausePlayback, playPlayback]);

  useEffect(() => {
    if (status.didJustFinish) {
      releasePlaybackOwnership();
    }
  }, [releasePlaybackOwnership, status.didJustFinish]);

  // Cleanup au démontage seulement — éviter de couper la lecture quand le player se stabilise.
  useEffect(() => {
    return () => {
      try {
        playerRef.current.pause();
      } catch {
        /* ignore */
      }
      releasePlaybackOwnership();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- démontage uniquement
  }, []);

  const durationSeconds = Math.max(0, Number(status.duration ?? player.duration ?? 0));
  const currentTimeSeconds = Math.max(0, Number(status.currentTime ?? player.currentTime ?? 0));
  const progress =
    durationSeconds > 0 ? Math.min(1, Math.max(0, currentTimeSeconds / durationSeconds)) : 0;
  const isLoaded = Boolean(
    status.isLoaded ||
      durationSeconds > 0 ||
      (typeof (player as { isLoaded?: boolean }).isLoaded === "boolean" &&
        (player as { isLoaded?: boolean }).isLoaded)
  );

  return {
    isPlaying: Boolean(status.playing ?? player.playing),
    isLoaded,
    isBuffering: Boolean(status.isBuffering),
    durationSeconds,
    currentTimeSeconds,
    progress,
    togglePlayback,
    pausePlayback,
  };
}
