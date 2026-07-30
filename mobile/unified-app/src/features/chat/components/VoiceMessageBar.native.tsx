import { useCallback, useMemo, useState } from "react";
import { Pressable, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Ionicons } from "@expo/vector-icons";
import { useChatVoicePlayer } from "../services/audioAdapter";
import {
  C_BUBBLE_OWN,
  VOICE_WAVEFORM_HEIGHT,
  voiceStyles as styles,
} from "./voiceMessageStyles";
import { buildVoiceWaveformHeights, formatVoiceDuration } from "./voiceWaveform";

type VoiceMessageBarProps = { uri: string; isOwn: boolean };

/**
 * Lecteur vocal style WhatsApp (horizontal) — sans avatar.
 * Natif via `expo-audio`.
 */
export function VoiceMessageBar({ uri, isOwn }: VoiceMessageBarProps) {
  const { isPlaying, progress, durationSeconds, currentTimeSeconds, togglePlayback } =
    useChatVoicePlayer(uri);
  const [busy, setBusy] = useState(false);
  const heights = useMemo(() => buildVoiceWaveformHeights(uri), [uri]);

  const toggle = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    try {
      await togglePlayback();
    } catch {
      /* ignore */
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

  return (
    <Pressable
      onPress={toggle}
      style={({ pressed }) => [
        styles.voiceRow,
        isOwn && styles.voiceRowOwn,
        pressed && styles.voiceRowPressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={
        isPlaying ? "Mettre le message vocal en pause" : "Lire le message vocal"
      }
    >
      <View style={[styles.playButton, isOwn && styles.playButtonOwn]}>
        <Ionicons name={isPlaying ? "pause" : "play"} size={18} color={iconColor} />
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
          ]}
        >
          {durationLabel}
        </AppText>
      </View>
    </Pressable>
  );
}
