import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import type {
  DashboardLiveActivityItem,
  DashboardLiveActivityTimeKind,
} from "../../dashboard/companyDashboardViewModel";
import { AppText } from "../../../../design/ui/AppText";
import { D } from "../../theme/companyDashboardTokens";
import { M, opsSurface } from "./dashboardMobileTokens";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

type Props = {
  items: DashboardLiveActivityItem[];
  onPressSeeAll?: () => void;
};

function feedIcon(item: DashboardLiveActivityItem): {
  name: keyof typeof Ionicons.glyphMap;
  bg: string;
  fg: string;
} {
  if (item.isDelayed || item.kind === "mission_delayed") {
    return { name: "warning", bg: "rgba(254, 226, 226, 0.9)", fg: D.danger };
  }
  switch (item.kind) {
    case "driver_available":
      return { name: "person", bg: "rgba(209, 250, 229, 0.9)", fg: D.brandDark };
    case "mission_active":
      return { name: "navigate", bg: "rgba(219, 234, 254, 0.95)", fg: D.inProgress };
    case "inbox_event":
      return { name: "notifications", bg: "rgba(237, 233, 254, 0.95)", fg: "#7C3AED" };
    case "network_alert":
      return { name: "pulse", bg: "rgba(254, 243, 199, 0.95)", fg: "#D97706" };
    case "empty_state":
      return { name: "checkmark-circle", bg: "rgba(241, 245, 249, 0.95)", fg: D.textMuted };
    default:
      break;
  }
  return { name: "radio-button-on", bg: "rgba(241, 245, 249, 0.95)", fg: D.textMuted };
}

function timeCaptionStyle(timeKind: DashboardLiveActivityTimeKind) {
  switch (timeKind) {
    case "instant":
      return s.timeInstant;
    case "scheduled":
      return s.timeScheduled;
    case "received_at":
      return s.timeReceived;
    case "day_summary":
    default:
      return s.timeSummary;
  }
}

export function DashboardLiveFeed({ items, onPressSeeAll }: Props) {
  if (items.length === 0) return null;

  return (
    <View accessibilityLabel="État opérationnel">
      <View style={opsSurface.sectionHead}>
        <AppText style={opsSurface.sectionTitle}>État opérationnel</AppText>
        {onPressSeeAll ? (
          <Pressable onPress={onPressSeeAll} hitSlop={8} accessibilityRole="button">
            <AppText style={opsSurface.sectionLink}>Tout</AppText>
          </Pressable>
        ) : null}
      </View>
      <View style={s.list}>
        {items.map((item, index) => {
          const icon = feedIcon(item);
          const caption = item.timeCaption || item.timeLabel;
          return (
            <View
              key={item.id}
              style={[s.row, index < items.length - 1 && s.rowBorder]}
              accessibilityRole="text"
              accessibilityLabel={`${item.message}. ${caption}`}
            >
              <View style={[s.iconWell, { backgroundColor: icon.bg }]}>
                <Ionicons name={icon.name} size={16} color={icon.fg} />
              </View>
              <View style={s.body}>
                <AppText
                  variant="caption"
                  style={[s.message, item.isDelayed && s.messageDanger]}
                  numberOfLines={2}
                >
                  {item.message}
                </AppText>
                {item.detail ? (
                  <AppText variant="caption" style={s.detail} numberOfLines={1}>
                    {item.detail}
                  </AppText>
                ) : null}
              </View>
              <AppText
                variant="caption"
                style={[s.time, timeCaptionStyle(item.timeKind)]}
                numberOfLines={2}
              >
                {caption}
              </AppText>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  list: {
    paddingBottom: 6,
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
    paddingHorizontal: M.padH,
    paddingVertical: M.padRow,
    minHeight: 44,
  },
  rowBorder: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.hairline,
  },
  iconWell: {
    width: M.feedIcon,
    height: M.feedIcon,
    borderRadius: M.feedIcon / 2,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    marginTop: 1,
  },
  body: { flex: 1, minWidth: 0, justifyContent: "center", gap: 2 },
  message: {
    fontSize: FONT_SIZE.px13,
    fontWeight: "600",
    color: D.text,
    lineHeight: 17,
    letterSpacing: -0.15,
  },
  messageDanger: {
    color: D.danger,
  },
  detail: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "500",
    color: D.textMuted,
    lineHeight: 14,
  },
  time: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "600",
    flexShrink: 0,
    maxWidth: 72,
    textAlign: "right",
    lineHeight: 13,
    marginTop: 1,
  },
  timeInstant: {
    color: D.brandDark,
  },
  timeScheduled: {
    color: D.inProgress,
  },
  timeReceived: {
    color: D.textMuted,
  },
  timeSummary: {
    color: D.textMuted,
    fontStyle: "italic",
  },
});
