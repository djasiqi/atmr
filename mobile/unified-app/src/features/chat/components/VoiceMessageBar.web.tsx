import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pressable, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Ionicons } from "@expo/vector-icons";
import {
  C_BUBBLE_OWN,
  VOICE_WAVEFORM_HEIGHT,
  voiceStyles as styles,
} from "./voiceMessageStyles";
import { buildVoiceWaveformHeights, formatVoiceDuration } from "./voiceWaveform";

type VoiceMessageBarProps = { uri: string; isOwn: boolean };

/**
 * Bundle web : lecture via `HTMLAudioElement` — layout WhatsApp horizontal, sans avatar.
 */
export function VoiceMessageBar({ uri, isOwn }: VoiceMessageBarProps) {
  const [playing, setPlaying] = useState(false);
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const heights = useMemo(() => buildVoiceWaveformHeights(uri), [uri]);

  useEffect(() => {
    return () => {
      const a = audioRef.current;
      audioRef.current = null;
      if (a) {
        a.pause();
        a.src = "";
      }
    };
  }, [uri]);

  const ensureAudio = useCallback(() => {
    if (typeof globalThis === "undefined" || typeof (globalThis as { Audio?: unknown }).Audio === "undefined") {
      return null;
    }
    if (!audioRef.current) {
      const Ctor = (globalThis as { Audio: new (s: string) => HTMLAudioElement }).Audio;
      const el = new Ctor(uri);
      el.addEventListener("ended", () => {
        setPlaying(false);
        setCurrentTimeSeconds(0);
      });
      el.addEventListener("loadedmetadata", () => {
        setDurationSeconds(Number.isFinite(el.duration) ? el.duration : 0);
      });
      el.addEventListener("timeupdate", () => {
        setCurrentTimeSeconds(el.currentTime);
      });
      audioRef.current = el;
    }
    return audioRef.current;
  }, [uri]);

  const [error, setError] = useState<string | null>(null);

  const toggle = useCallback(() => {
    const el = ensureAudio();
    if (!el) {
      setError("Impossible de lire ce message vocal.");
      return;
    }
    setError(null);
    if (playing) {
      el.pause();
      setPlaying(false);
      return;
    }
    void el
      .play()
      .then(() => setPlaying(true))
      .catch(() => {
        setPlaying(false);
        setError("Impossible de lire ce message vocal.");
      });
  }, [ensureAudio, playing]);

  const progress =
    durationSeconds > 0 ? Math.min(1, Math.max(0, currentTimeSeconds / durationSeconds)) : 0;
  const playedIndex = Math.floor(progress * heights.length);
  const displaySeconds =
    playing || currentTimeSeconds > 0.05
      ? Math.max(0, durationSeconds - currentTimeSeconds)
      : durationSeconds;
  const durationLabel = formatVoiceDuration(displaySeconds);
  const barPlayed = isOwn ? "rgba(255,255,255,0.95)" : C_BUBBLE_OWN;
  const barIdle = isOwn ? "rgba(255,255,255,0.38)" : "rgba(148,163,184,0.85)";
  const scrubColor = isOwn ? "#FFFFFF" : C_BUBBLE_OWN;
  const iconColor = isOwn ? "#ECFDF5" : C_BUBBLE_OWN;

  return (
    <Pressable
      onPress={toggle}
      style={({ pressed }) => [
        styles.voiceRow,
        isOwn && styles.voiceRowOwn,
        pressed && styles.voiceRowPressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={playing ? "Mettre le message vocal en pause" : "Lire le message vocal"}
    >
      <View style={[styles.playButton, isOwn && styles.playButtonOwn]}>
        <Ionicons name={playing ? "pause" : "play"} size={18} color={iconColor} />
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
          {error ?? durationLabel}
        </AppText>
      </View>
    </Pressable>
  );
}
