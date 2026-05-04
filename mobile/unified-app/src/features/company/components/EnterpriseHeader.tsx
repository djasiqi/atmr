import type { ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";
import { AppText } from "../../../design/ui/AppText";
import { E } from "../theme/enterpriseOpsTheme";
import { createShadow } from "../../../styles/shadowStyles";

dayjs.locale("fr");

type EnterpriseHeaderProps = {
  date: string;
  mode: string | null;
  realtimeStatus: string;
  /** Icône cloche / autre, à droite (ex. boîte entreprise). */
  trailing?: ReactNode;
};

function formatMode(mode: string | null): string {
  if (mode === "manual") return "Manuel";
  if (mode === "semi_auto") return "Semi-auto";
  if (mode === "fully_auto") return "Plein auto";
  if (mode == null || mode === "") return "—";
  return mode;
}

function formatRealtime(s: string): string {
  if (s === "healthy") return "ok";
  if (s === "idle") return "inactif";
  return s;
}

function formatDayLine(isoDate: string): string {
  const d = dayjs(isoDate);
  if (!d.isValid()) return isoDate;
  return d.format("dddd D MMMM");
}

const headerCardShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

export function EnterpriseHeader({ date, mode, realtimeStatus, trailing }: EnterpriseHeaderProps) {
  const dayLine = formatDayLine(date);
  return (
    <View style={s.header}>
      <View style={s.headerTop}>
        <View style={s.headerLeft}>
          <View style={s.headerIconWrap}>
            <Ionicons name="car-outline" size={20} color={E.BRAND} />
          </View>
          <View style={s.titleBlock}>
            <AppText variant="sectionTitle" style={s.headerTitle} accessibilityRole="header">
              Courses
            </AppText>
            <AppText
              variant="caption"
              style={s.headerDate}
              accessibilityLabel={`Jour sélectionné : ${dayLine}`}
            >
              {dayLine}
            </AppText>
          </View>
        </View>
        {trailing != null ? <View style={s.headerRight}>{trailing}</View> : null}
      </View>
      <AppText variant="caption" style={s.meta} numberOfLines={2}>
        <AppText variant="caption" style={s.metaKey}>
          Journée{" "}
        </AppText>
        <AppText variant="caption" style={s.metaValue}>
          {date}
        </AppText>
        <AppText variant="caption" style={s.metaSep}>
          {" "}
          ·{" "}
        </AppText>
        <AppText variant="caption" style={s.metaKey}>
          Mode{" "}
        </AppText>
        <AppText variant="caption" style={s.metaValue}>
          {formatMode(mode)}
        </AppText>
        <AppText variant="caption" style={s.metaSep}>
          {" "}
          ·{" "}
        </AppText>
        <AppText variant="caption" style={s.metaKey}>
          Réseau{" "}
        </AppText>
        <AppText variant="caption" style={s.metaValue}>
          {formatRealtime(realtimeStatus)}
        </AppText>
      </AppText>
    </View>
  );
}

const s = StyleSheet.create({
  header: {
    backgroundColor: E.CARD,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: E.BORDER,
    marginBottom: 0,
    ...headerCardShadow,
  },
  headerTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 10 },
  headerLeft: { flexDirection: "row", alignItems: "center", gap: 10, flex: 1, minWidth: 0 },
  headerRight: { flexDirection: "row", alignItems: "center", gap: 8, flexShrink: 0 },
  titleBlock: { flex: 1, minWidth: 0 },
  headerIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  headerTitle: { color: E.TEXT, letterSpacing: -0.2 },
  headerDate: {
    color: E.TEXT_SEC,
    marginTop: 1,
    textTransform: "capitalize" as const,
  },
  meta: { color: E.TEXT_MUTED, marginTop: 8 },
  metaSep: { color: E.TEXT_MUTED },
  metaKey: { color: E.TEXT_MUTED, fontWeight: "600" as const },
  metaValue: { color: E.TEXT_MUTED, fontWeight: "400" as const },
});
