import { useCallback, useEffect, useRef, useState } from "react";
import { Pressable, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { C_BUBBLE_OWN, voiceStyles as styles } from "./voiceMessageStyles";

type VoiceMessageBarProps = { uri: string; isOwn: boolean };

/**
 * Bundle web : lecture via `HTMLAudioElement` (aucun `expo-av`).
 */
export function VoiceMessageBar({ uri, isOwn }: VoiceMessageBarProps) {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

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

  const toggle = useCallback(() => {
    if (typeof globalThis === "undefined" || typeof (globalThis as { Audio?: unknown }).Audio === "undefined") {
      return;
    }
    const Ctor = (globalThis as { Audio: new (s: string) => HTMLAudioElement }).Audio;
    if (!audioRef.current) {
      const el = new Ctor(uri);
      el.addEventListener("ended", () => setPlaying(false));
      audioRef.current = el;
    }
    const el = audioRef.current;
    if (!el) return;
    if (playing) {
      el.pause();
      setPlaying(false);
    } else {
      void el.play()
        .then(() => setPlaying(true))
        .catch(() => setPlaying(false));
    }
  }, [playing, uri]);

  return (
    <Pressable
      onPress={toggle}
      style={({ pressed }) => [styles.voiceRow, isOwn && styles.voiceRowOwn, pressed && styles.voiceRowPressed]}
      accessibilityRole="button"
      accessibilityLabel={playing ? "Mettre le message vocal en pause" : "Lire le message vocal"}
    >
      <View style={[styles.voiceIconBox, isOwn && styles.voiceIconBoxOwn]}>
        <Ionicons name={playing ? "pause" : "play"} size={20} color={isOwn ? "#ecfdf5" : C_BUBBLE_OWN} />
      </View>
      <View style={styles.voiceTextBlock}>
        <Text style={[styles.voiceTitle, isOwn && styles.voiceTitleOwn]} numberOfLines={1}>
          Message vocal
        </Text>
        <Text style={[styles.voiceMeta, isOwn && styles.voiceMetaOwn]}>Appuyez pour écouter</Text>
      </View>
    </Pressable>
  );
}
