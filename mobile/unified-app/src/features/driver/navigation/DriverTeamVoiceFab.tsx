import { Platform, Pressable, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import {
  useDriverTeamVoiceBroadcast,
  type DriverTeamVoiceFeedback,
} from "./useDriverTeamVoiceBroadcast";

const C = {
  brand: "#00796B",
  recording: "#DC2626",
} as const;

const PRESSABLE_WEB_SUPPRESS_SQUARE_HALO = Platform.select({
  web: {
    cursor: "pointer",
    outlineWidth: 0,
    outlineStyle: "none",
    // @ts-expect-error RN web
    WebkitTapHighlightColor: "transparent",
  } as const,
  default: undefined,
});

type DriverTeamVoiceFabProps = {
  onFeedback?: (feedback: DriverTeamVoiceFeedback | null) => void;
};

/** Micro central : envoi vocal direct vers le canal équipe de l'entreprise. */
export function DriverTeamVoiceFab({ onFeedback }: DriverTeamVoiceFabProps) {
  const { disabled, isRecording, voiceBusy, handlePress } = useDriverTeamVoiceBroadcast({
    onFeedback,
  });

  const a11yLabel = disabled
    ? "Messages vocaux indisponibles sur le web"
    : voiceBusy
      ? "Envoi du message vocal en cours"
      : isRecording
        ? "Appuyez pour envoyer au canal équipe"
        : "Appuyez pour enregistrer un message vocal pour le canal équipe";

  return (
    <Pressable
      onPress={handlePress}
      disabled={disabled || voiceBusy}
      accessibilityLabel={a11yLabel}
      accessibilityRole="button"
      accessibilityState={{ disabled: disabled || voiceBusy, selected: isRecording }}
      android_ripple={
        Platform.OS === "android"
          ? { color: "rgba(255, 255, 255, 0.35)", borderless: true }
          : undefined
      }
      style={({ pressed }) => [
        styles.fabOuter,
        isRecording && styles.fabRecording,
        voiceBusy && styles.fabBusy,
        pressed && !voiceBusy && !isRecording && styles.fabOuterPressed,
        Platform.OS === "web" ? PRESSABLE_WEB_SUPPRESS_SQUARE_HALO : null,
      ]}
    >
      <Ionicons
        name={voiceBusy ? "hourglass-outline" : isRecording ? "stop" : "mic"}
        size={22}
        color="#FFFFFF"
      />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  fabOuter: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.brand,
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "center",
    ...Platform.select({
      web: {
        boxShadow: "0 1px 4px rgba(10, 58, 52, 0.2)",
      } as const,
      default: {
        elevation: 2,
        shadowColor: "#163A34",
        shadowOpacity: 0.2,
        shadowOffset: { width: 0, height: 1 },
        shadowRadius: 2,
      },
    }),
  },
  fabRecording: {
    backgroundColor: C.recording,
  },
  fabBusy: {
    opacity: 0.85,
  },
  fabOuterPressed: {
    opacity: 0.9,
  },
});
