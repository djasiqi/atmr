import { StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

export function formatChatDayLabel(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  const now = new Date();
  const msg = new Date(d);
  const sameDay =
    now.getFullYear() === msg.getFullYear() &&
    now.getMonth() === msg.getMonth() &&
    now.getDate() === msg.getDate();
  if (sameDay) return "Aujourd'hui";
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  const isYesterday =
    yesterday.getFullYear() === msg.getFullYear() &&
    yesterday.getMonth() === msg.getMonth() &&
    yesterday.getDate() === msg.getDate();
  if (isYesterday) return "Hier";
  return msg.toLocaleDateString("fr-FR", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
}

export function dayKeyFromIso(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  const msg = new Date(d);
  return `${msg.getFullYear()}-${msg.getMonth()}-${msg.getDate()}`;
}

type Props = { label: string; density?: "default" | "compact" };

export function ChatDateSeparator({ label, density = "default" }: Props) {
  const compact = density === "compact";
  return (
    <View style={[styles.wrap, compact && styles.wrapCompact]}>
      <View style={[styles.pill, compact && styles.pillCompact]}>
        <AppText variant="caption" style={styles.text}>
          {label}
        </AppText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { alignItems: "center", marginVertical: 12 },
  wrapCompact: { marginVertical: 5 },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: "#E5E7EB",
  },
  pillCompact: {
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 12,
  },
  text: {
    color: "#4B5563",
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
    textTransform: "capitalize",
  },
});
