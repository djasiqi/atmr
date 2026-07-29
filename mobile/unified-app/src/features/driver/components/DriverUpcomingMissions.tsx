import { useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { createShadow } from "../../../styles/shadowStyles";
import { D, dashboardCardShadow } from "../theme/driverDashboardTheme";
import type { DriverMission } from "../types";
import { getMissionClientDisplayName } from "../domain/missionDisplay";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const cardShadow = createShadow(dashboardCardShadow);

const SWISS_TZ = "Europe/Zurich";

function formatWhen(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (!Number.isFinite(d.getTime())) return "—";
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function conciseAddress(s: string | null | undefined, maxLen = 42): string {
  const t = s?.trim() ?? "";
  if (!t) return "—";
  const head = t.split(",")[0]?.trim() || t;
  if (head.length <= maxLen) return head;
  return `${head.slice(0, Math.max(0, maxLen - 1))}…`;
}

type Props = {
  missions: DriverMission[];
  onOpenMission: (missionId: number) => void;
  onOpenAll: () => void;
};

export function DriverUpcomingMissions({ missions, onOpenMission, onOpenAll }: Props) {
  const [expanded, setExpanded] = useState(false);
  const { isVeryLargeText } = useAccessibilityScale();

  if (missions.length === 0) return null;

  return (
    <View style={styles.card}>
      <Pressable
        onPress={() => setExpanded((v) => !v)}
        accessibilityRole="button"
        accessibilityState={{ expanded }}
        accessibilityLabel={`Prochaines missions, ${missions.length} à venir`}
        style={({ pressed }) => [styles.headerRow, pressed && styles.pressed]}
      >
        <View style={styles.headerLeft}>
          <View style={styles.iconWrap} accessibilityElementsHidden>
            <Ionicons name="time-outline" size={15} color={D.brand} />
          </View>
          <AppText variant="sectionTitle" style={styles.title}>
            Prochaines missions
          </AppText>
          <View style={styles.countBadge}>
            <AppText variant="caption" style={styles.countText}>
              {missions.length}
            </AppText>
          </View>
        </View>
        <Ionicons
          name={expanded ? "chevron-up" : "chevron-down"}
          size={18}
          color={D.textMuted}
          accessibilityElementsHidden
        />
      </Pressable>

      {expanded ? (
        <View style={styles.body}>
          {missions.map((m, index) => {
            const scheduledRaw = (m.scheduled_time ?? m.scheduled_at) as string | null | undefined;
            const clientName = getMissionClientDisplayName(m);
            return (
              <Pressable
                key={m.id}
                onPress={() => onOpenMission(m.id)}
                accessibilityRole="button"
                accessibilityLabel={`Ouvrir mission ${m.id}`}
                style={({ pressed }) => [
                  styles.missionRow,
                  index < missions.length - 1 && styles.missionRowSep,
                  pressed && styles.pressed,
                ]}
              >
                <AppText variant="label" style={styles.when} numberOfLines={isVeryLargeText ? undefined : 1}>
                  {formatWhen(scheduledRaw)}
                </AppText>
                <AppText variant="label" style={styles.client} numberOfLines={isVeryLargeText ? undefined : 1}>
                  {clientName}
                </AppText>
                <AppText variant="caption" style={styles.address} numberOfLines={isVeryLargeText ? undefined : 2}>
                  {conciseAddress(m.pickup_location as string | null | undefined, isVeryLargeText ? 120 : 42)}
                </AppText>
              </Pressable>
            );
          })}
          <Pressable
            onPress={onOpenAll}
            accessibilityRole="button"
            accessibilityLabel="Voir toutes les missions"
            style={({ pressed }) => [styles.linkRow, pressed && styles.pressed]}
          >
            <AppText variant="label" style={styles.linkText}>
              Voir toutes les missions
            </AppText>
            <Ionicons name="chevron-forward" size={16} color={D.brand} />
          </Pressable>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: D.cardBg,
    borderRadius: D.controlRadius,
    borderWidth: 1,
    borderColor: D.cardBorder,
    overflow: "visible",
    ...cardShadow,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 14,
    paddingVertical: 13,
    minHeight: 48,
  },
  headerLeft: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    minWidth: 0,
  },
  iconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    color: D.text,
    fontSize: FONT_SIZE.px15,
    fontWeight: "700",
    flexShrink: 1,
  },
  countBadge: {
    minWidth: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: "rgba(0, 121, 107, 0.1)",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 6,
  },
  countText: {
    color: D.brand,
    fontWeight: "800",
    fontSize: FONT_SIZE.px11,
  },
  body: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: D.cardBorder,
    paddingHorizontal: 14,
    paddingBottom: 10,
  },
  missionRow: {
    gap: 2,
    paddingVertical: 10,
  },
  missionRowSep: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: D.cardBorder,
  },
  when: {
    color: D.brand,
    fontWeight: "800",
    fontSize: FONT_SIZE.px12,
    flexShrink: 1,
    minWidth: 0,
  },
  client: {
    color: D.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px14,
    flexShrink: 1,
    minWidth: 0,
  },
  address: {
    color: D.textSub,
    fontWeight: "500",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    flexShrink: 1,
    minWidth: 0,
  },
  linkRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    minHeight: 42,
    paddingTop: 4,
  },
  linkText: {
    color: D.brand,
    fontWeight: "800",
    fontSize: FONT_SIZE.px13,
  },
  pressed: { opacity: 0.88 },
});
