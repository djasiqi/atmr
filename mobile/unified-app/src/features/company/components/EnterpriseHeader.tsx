import type { ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";
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
            <Text style={s.headerTitle} accessibilityRole="header">
              Courses
            </Text>
            <Text style={s.headerDate} accessibilityLabel={`Jour sélectionné : ${dayLine}`}>
              {dayLine}
            </Text>
          </View>
        </View>
        {trailing != null ? <View style={s.headerRight}>{trailing}</View> : null}
      </View>
      <Text style={s.meta} numberOfLines={2}>
        <Text style={s.metaKey}>Journée </Text>
        <Text style={s.metaValue}>{date}</Text>
        <Text style={s.meta}> · </Text>
        <Text style={s.metaKey}>Mode </Text>
        <Text style={s.metaValue}>{formatMode(mode)}</Text>
        <Text style={s.meta}> · </Text>
        <Text style={s.metaKey}>Réseau </Text>
        <Text style={s.metaValue}>{formatRealtime(realtimeStatus)}</Text>
      </Text>
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
  headerTitle: { fontSize: 17, fontWeight: "700" as const, color: E.TEXT, letterSpacing: -0.2 },
  headerDate: {
    color: E.TEXT_SEC,
    fontSize: 12,
    marginTop: 1,
    textTransform: "capitalize" as const,
  },
  meta: { color: E.TEXT_MUTED, fontSize: 12, lineHeight: 17.5, marginTop: 8 },
  metaKey: { color: E.TEXT_MUTED, fontSize: 12, fontWeight: "600" as const },
  metaValue: { color: E.TEXT, fontSize: 12, fontWeight: "700" as const },
});
