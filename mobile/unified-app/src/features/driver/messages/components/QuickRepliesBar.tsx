import { useMemo } from "react";
import { ScrollView, Pressable, StyleSheet } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { useDriverMissionsQuery } from "../../hooks";
import { D } from "../../theme/driverDashboardTheme";
import type { ChannelQuickRepliesMode } from "../channelQuickReplies";
import {
  buildTeamQuickReplies,
  selectActiveDriverMission,
} from "../teamQuickReplies";
import { QUICK_REPLY_TEMPLATES } from "../types";

type QuickTemplate = { id: string; label: string; content: string };

type Props = {
  onSelect: (content: string) => void;
  mode?: ChannelQuickRepliesMode;
};

export function QuickRepliesBar({ onSelect, mode = "off" }: Props) {
  const missionsQuery = useDriverMissionsQuery();
  const activeMission = useMemo(
    () => selectActiveDriverMission(missionsQuery.data),
    [missionsQuery.data]
  );
  const templates: readonly QuickTemplate[] = useMemo(() => {
    if (mode === "off") return [];
    if (mode === "team-mission") return buildTeamQuickReplies(activeMission);
    return QUICK_REPLY_TEMPLATES;
  }, [mode, activeMission]);

  if (templates.length === 0) return null;

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={[styles.row, styles.rowTeam]}
      keyboardShouldPersistTaps="handled"
    >
      {templates.map((item) => (
        <Pressable
          key={item.id}
          style={({ pressed }) => [styles.chip, styles.chipTeam, pressed && styles.chipPressed]}
          onPress={() => onSelect(item.content)}
          accessibilityRole="button"
          accessibilityLabel={`Envoyer : ${item.content}`}
        >
          <AppText variant="caption" style={styles.chipText}>
            {item.label}
          </AppText>
        </Pressable>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  row: { gap: 8, paddingVertical: 6 },
  rowTeam: {
    paddingHorizontal: 0,
    paddingVertical: 2,
    backgroundColor: "transparent",
  },
  chip: {
    backgroundColor: "#ecfdf5",
    borderColor: "#a7f3d0",
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  chipTeam: {
    backgroundColor: D.pageBg,
    borderColor: "#E5E7EB",
  },
  chipPressed: { opacity: 0.88 },
  chipText: { color: "#065f46", fontWeight: "600" },
});
