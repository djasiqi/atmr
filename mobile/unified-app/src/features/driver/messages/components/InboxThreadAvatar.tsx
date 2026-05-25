import { Image, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import type { ThreadDisplayLines } from "../inboxDisplay";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
const BRAND_LOGO = require("../../../../../assets/images/lirie-logo-color.png");

type Props = {
  lines: ThreadDisplayLines;
  titleFallback: string;
};

export function InboxThreadAvatar({ lines, titleFallback }: Props) {
  const initials = titleFallback
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? "")
    .join("");

  if (lines.avatarKind === "support") {
    return (
      <View style={[styles.circle, styles.supportBg]}>
        <Image source={BRAND_LOGO} style={styles.logo} resizeMode="contain" />
      </View>
    );
  }

  if (lines.avatarKind === "mission") {
    return (
      <View style={[styles.circle, styles.missionBg]}>
        <Ionicons name="car-outline" size={22} color="#fff" />
      </View>
    );
  }

  if (lines.avatarKind === "dispatch") {
    return (
      <View style={[styles.circle, styles.dispatchBg]}>
        <Ionicons name="business-outline" size={22} color="#fff" />
      </View>
    );
  }

  if (lines.avatarKind === "group") {
    return (
      <View style={[styles.circle, styles.groupBg]}>
        <Ionicons name="people-outline" size={22} color="#fff" />
      </View>
    );
  }

  return (
    <View style={[styles.circle, styles.personBg]}>
      <AppText variant="body" style={styles.initials}>
        {initials || "?"}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  circle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  missionBg: { backgroundColor: M.BRAND_DARK },
  dispatchBg: { backgroundColor: M.BRAND },
  groupBg: { backgroundColor: "#64748B" },
  supportBg: { backgroundColor: M.PAGE_BG },
  personBg: { backgroundColor: M.TEXT_MUTED },  logo: { width: 36, height: 36 },
  initials: { color: "#fff", fontWeight: "700", fontSize: FONT_SIZE.px16 },
});
