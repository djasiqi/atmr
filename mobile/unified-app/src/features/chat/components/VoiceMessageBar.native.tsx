import { useEffect, useRef, useState } from "react";
import { Pressable, Text, View } from "react-native";
import { Audio, type AVPlaybackStatusSuccess } from "expo-av";
import { Ionicons } from "@expo/vector-icons";
import { C_BUBBLE_OWN, voiceStyles as styles } from "./voiceMessageStyles";

type VoiceMessageBarProps = { uri: string; isOwn: boolean };

/**
 * Lecteur audio (fichier local / distant) — natif, via expo-av.
 */
export function VoiceMessageBar({ uri, isOwn }: VoiceMessageBarProps) {
  const soundRef = useRef<Audio.Sound | null>(null);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    return () => {
      void (async () => {
        if (soundRef.current) {
          try {
            await soundRef.current.unloadAsync();
          } catch {
            /* ignore */
          }
        }
        soundRef.current = null;
      })();
    };
  }, [uri]);

  const toggle = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
      if (!soundRef.current) {
        const { sound } = await Audio.Sound.createAsync(
          { uri },
          { shouldPlay: false },
          (status) => {
            if (!status.isLoaded) return;
            const s = status as AVPlaybackStatusSuccess;
            if (s.didJustFinish) setPlaying(false);
          }
        );
        soundRef.current = sound;
      }
      const current = soundRef.current;
      const status = await current.getStatusAsync();
      if (status.isLoaded && status.isPlaying) {
        await current.pauseAsync();
        setPlaying(false);
      } else if (status.isLoaded) {
        const duration = status.durationMillis ?? 0;
        const pos = status.positionMillis ?? 0;
        const atEnd = duration > 0 && pos >= duration - 80;
        if (atEnd) {
          await current.setPositionAsync(0);
        }
        await current.playAsync();
        setPlaying(true);
      }
    } catch {
      setPlaying(false);
    } finally {
      setBusy(false);
    }
  };

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
