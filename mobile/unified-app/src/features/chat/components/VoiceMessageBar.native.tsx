import { useCallback, useEffect, useMemo, useState } from "react";
import { ActivityIndicator, Pressable, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Ionicons } from "@expo/vector-icons";
import { useChatVoicePlayer } from "../services/audioAdapter";
import { resolvePlayableChatAudioUri } from "../services/resolvePlayableChatAudio";
import {
  C_BUBBLE_OWN,
  VOICE_WAVEFORM_HEIGHT,
  voiceStyles as styles,
} from "./voiceMessageStyles";
import { buildVoiceWaveformHeights, formatVoiceDuration } from "./voiceWaveform";

type VoiceMessageBarProps = {
  uri: string;
  isOwn: boolean;
  /** Id DB du message — requis pour télécharger `/uploads/chat` (privé SEC-06). */
  messageId?: string | number | null;
};

type ReadyBarProps = {
  uri: string;
  isOwn: boolean;
  heights: number[];
};

function ReadyVoiceMessageBar({ uri, isOwn, heights }: ReadyBarProps) {
  const {
    isPlaying,
    isLoaded,
    isBuffering,
    progress,
    durationSeconds,
    currentTimeSeconds,
    togglePlayback,
  } = useChatVoicePlayer(uri);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await togglePlayback();
      if (!result.ok) {
        setError("Impossible de lire ce message vocal.");
      }
    } catch {
      setError("Impossible de lire ce message vocal.");
    } finally {
      setBusy(false);
    }
  }, [busy, togglePlayback]);

  const playedIndex = Math.floor(progress * heights.length);
  const displaySeconds =
    isPlaying || currentTimeSeconds > 0.05
      ? Math.max(0, durationSeconds - currentTimeSeconds)
      : durationSeconds;
  const durationLabel = formatVoiceDuration(displaySeconds);
  const barPlayed = isOwn ? "rgba(255,255,255,0.95)" : C_BUBBLE_OWN;
  const barIdle = isOwn ? "rgba(255,255,255,0.38)" : "rgba(148,163,184,0.85)";
  const scrubColor = isOwn ? "#FFFFFF" : C_BUBBLE_OWN;
  const iconColor = isOwn ? "#ECFDF5" : C_BUBBLE_OWN;
  const showSpinner = busy || (isBuffering && !isPlaying);

  return (
    <Pressable
      onPress={toggle}
      disabled={busy}
      style={({ pressed }) => [
        styles.voiceRow,
        isOwn && styles.voiceRowOwn,
        pressed && styles.voiceRowPressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={
        isPlaying ? "Mettre le message vocal en pause" : "Lire le message vocal"
      }
      accessibilityState={{ disabled: busy, busy: showSpinner }}
    >
      <View style={[styles.playButton, isOwn && styles.playButtonOwn]}>
        {showSpinner ? (
          <ActivityIndicator size="small" color={iconColor} />
        ) : (
          <Ionicons name={isPlaying ? "pause" : "play"} size={18} color={iconColor} />
        )}
      </View>

      <View style={styles.waveColumn}>
        <View style={styles.waveTrack}>
          {heights.map((h, index) => (
            <View
              key={`bar-${index}`}
              style={[
                styles.waveBar,
                {
                  height: Math.max(3, h * VOICE_WAVEFORM_HEIGHT),
                  backgroundColor: index <= playedIndex ? barPlayed : barIdle,
                },
              ]}
            />
          ))}
          <View
            pointerEvents="none"
            style={[
              styles.scrubDot,
              {
                left: `${Math.min(100, Math.max(0, progress * 100))}%`,
                backgroundColor: scrubColor,
              },
            ]}
          />
        </View>
        <AppText
          variant="caption"
          scaleRole="chrome"
          style={[
            styles.durationText,
            isOwn ? styles.durationOwn : styles.durationIn,
            error ? styles.durationError : null,
            error && isOwn ? styles.durationErrorOwn : null,
          ]}
        >
          {error ?? (durationSeconds > 0 || isLoaded ? durationLabel : "…")}
        </AppText>
      </View>
    </Pressable>
  );
}

/**
 * Lecteur vocal style WhatsApp (horizontal) — sans avatar.
 * Natif via `expo-audio` après résolution auth des uploads chat privés.
 */
export function VoiceMessageBar({ uri, isOwn, messageId }: VoiceMessageBarProps) {
  const heights = useMemo(() => buildVoiceWaveformHeights(uri), [uri]);
  const [playableUri, setPlayableUri] = useState<string | null>(null);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [resolving, setResolving] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setResolving(true);
    setResolveError(null);
    setPlayableUri(null);
    void resolvePlayableChatAudioUri({ uri, messageId })
      .then((next) => {
        if (!cancelled) setPlayableUri(next);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message =
          err instanceof Error && err.message.trim()
            ? err.message.trim()
            : "Impossible de charger ce message vocal.";
        setResolveError(message);
      })
      .finally(() => {
        if (!cancelled) setResolving(false);
      });
    return () => {
      cancelled = true;
    };
  }, [uri, messageId]);

  if (resolving || (!playableUri && !resolveError)) {
    const iconColor = isOwn ? "#ECFDF5" : C_BUBBLE_OWN;
    return (
      <View style={[styles.voiceRow, isOwn && styles.voiceRowOwn]}>
        <View style={[styles.playButton, isOwn && styles.playButtonOwn]}>
          <ActivityIndicator size="small" color={iconColor} />
        </View>
        <View style={styles.waveColumn}>
          <View style={styles.waveTrack}>
            {heights.map((h, index) => (
              <View
                key={`bar-loading-${index}`}
                style={[
                  styles.waveBar,
                  {
                    height: Math.max(3, h * VOICE_WAVEFORM_HEIGHT),
                    backgroundColor: isOwn
                      ? "rgba(255,255,255,0.38)"
                      : "rgba(148,163,184,0.85)",
                  },
                ]}
              />
            ))}
          </View>
          <AppText
            variant="caption"
            scaleRole="chrome"
            style={[styles.durationText, isOwn ? styles.durationOwn : styles.durationIn]}
          >
            …
          </AppText>
        </View>
      </View>
    );
  }

  if (resolveError || !playableUri) {
    return (
      <View style={[styles.voiceRow, isOwn && styles.voiceRowOwn]}>
        <View style={[styles.playButton, isOwn && styles.playButtonOwn]}>
          <Ionicons
            name="alert-circle-outline"
            size={18}
            color={isOwn ? "#ECFDF5" : C_BUBBLE_OWN}
          />
        </View>
        <View style={styles.waveColumn}>
          <AppText
            variant="caption"
            scaleRole="chrome"
            style={[
              styles.durationText,
              styles.durationError,
              isOwn ? styles.durationErrorOwn : null,
            ]}
          >
            {resolveError ?? "Impossible de charger ce message vocal."}
          </AppText>
        </View>
      </View>
    );
  }

  return <ReadyVoiceMessageBar uri={playableUri} isOwn={isOwn} heights={heights} />;
}
