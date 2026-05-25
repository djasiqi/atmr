import { useCallback, useState } from "react";
import { Pressable, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import { Ionicons } from "@expo/vector-icons";
import { useChatVoicePlayer } from "../services/audioAdapter";
import { C_BUBBLE_OWN, voiceStyles as styles } from "./voiceMessageStyles";

type VoiceMessageBarProps = { uri: string; isOwn: boolean };

/**
 * Lecteur audio (fichier local / distant) — natif, via `expo-audio` (adaptateur interne).
 */
export function VoiceMessageBar({ uri, isOwn }: VoiceMessageBarProps) {
  const { isPlaying, togglePlayback } = useChatVoicePlayer(uri);
  const [busy, setBusy] = useState(false);

  const toggle = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    try {
      await togglePlayback();
    } catch {
      /* ignore : erreurs déjà gérées dans l’adaptateur */
    } finally {
      setBusy(false);
    }
  }, [busy, togglePlayback]);

  return (
    <Pressable
      onPress={toggle}
      style={({ pressed }) => [styles.voiceRow, isOwn && styles.voiceRowOwn, pressed && styles.voiceRowPressed]}
      accessibilityRole="button"
      accessibilityLabel={isPlaying ? "Mettre le message vocal en pause" : "Lire le message vocal"}
    >
      <View style={[styles.voiceIconBox, isOwn && styles.voiceIconBoxOwn]}>
        <Ionicons name={isPlaying ? "pause" : "play"} size={20} color={isOwn ? "#ecfdf5" : C_BUBBLE_OWN} />
      </View>
      <View style={styles.voiceTextBlock}>
        <AppText
          variant="label"
          numberOfLines={1}
          style={isOwn ? styles.voiceTitleOwn : undefined}
        >
          Message vocal
        </AppText>
        <AppText
          variant="caption"
          style={[styles.voiceMetaSpacing, isOwn && styles.voiceMetaOwn]}
        >
          Appuyez pour écouter
        </AppText>
      </View>
    </Pressable>
  );
}
