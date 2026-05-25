import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";

type Props = {
  content: string;
  timestamp: string;
  senderName?: string | null;
  variant?: "default" | "team";
};

function formatTime(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  return new Date(d).toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function SystemMessageRow({
  content,
  timestamp,
  senderName,
  variant = "default",
}: Props) {
  const time = formatTime(timestamp);
  const label = senderName?.trim() || "Système";

  if (variant === "team") {
    return (
      <View style={styles.teamWrap} accessibilityRole="text">
        <View style={styles.teamBox}>
          <Ionicons
            name={senderName ? "person-outline" : "settings-outline"}
            size={16}
            color={M.BRAND}
            style={styles.teamIcon}
          />
          <AppText variant="caption" style={styles.teamText}>
            <AppText variant="caption" style={styles.teamBold}>
              {label}
            </AppText>
            {time ? ` ${time}` : ""} — {content}
          </AppText>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.wrap} accessibilityRole="text">
      <View style={styles.line} />
      {senderName ? (
        <AppText variant="caption" style={styles.reporter}>
          {label}
        </AppText>
      ) : null}
      <AppText variant="caption" style={styles.content}>
        {content}
      </AppText>
      <AppText variant="caption" style={styles.time}>
        {time}
      </AppText>
      <View style={styles.line} />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: "center",
    marginVertical: 8,
    gap: 4,
    paddingHorizontal: 12,
  },
  line: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: "#d1d5db",
    alignSelf: "stretch",
  },
  content: {
    color: "#4b5563",
    textAlign: "center",
    fontWeight: "600",
  },
  reporter: {
    color: "#047857",
    textAlign: "center",
    fontWeight: "700",
  },
  time: { color: "#9ca3af" },
  teamWrap: {
    alignItems: "center",
    marginVertical: 5,
    paddingHorizontal: 12,
  },
  teamBox: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 6,
    backgroundColor: "#ECFDF5",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#A7F3D0",
    maxWidth: "92%",
  },
  teamIcon: { marginTop: 1 },
  teamText: { flex: 1, color: "#065F46", lineHeight: 18 },
  teamBold: { fontWeight: "700", color: "#047857" },
});
