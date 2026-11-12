import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import {
  driverCourseBadgeTone,
  driverCourseStatusLabel,
} from "../domain/driverCourseLateness";

type Props = {
  status: string | null | undefined;
  late: boolean;
};

/** Un seul badge : le libellé métier reste, le rouge signale un retard de prise en charge. */
export function DriverCourseStatusBadge({ status, late }: Props) {
  const label = driverCourseStatusLabel(status);
  const tone = driverCourseBadgeTone(status, late);
  return (
    <View
      testID="driver-course-status-badge"
      accessibilityRole="text"
      accessibilityLabel={late ? `${label}. En retard` : label}
      style={[styles.badge, { backgroundColor: tone.background }]}
    >
      <Ionicons name={tone.icon} size={12} color={tone.foreground} />
      <AppText
        variant="label"
        scaleRole="chrome"
        numberOfLines={2}
        style={[styles.text, { color: tone.foreground }]}
      >
        {label}
      </AppText>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-end",
    flexShrink: 1,
    maxWidth: "100%",
    gap: 4,
    minHeight: 24,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  text: {
    flexShrink: 1,
    fontWeight: "700",
  },
});
