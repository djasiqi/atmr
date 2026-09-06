import { Pressable, StyleSheet, View } from "react-native";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { EnterpriseBottomSheet } from "../EnterpriseBottomSheet";
import type { LiveCoverageRow } from "./liveGpsCoverage";

type Props = {
  visible: boolean;
  onClose: () => void;
  summary: string;
  rows: LiveCoverageRow[];
  onSelectDriver?: (driverId: number) => void;
};

/** Feuille couverture GPS — liste présence, distincte de la barre opérationnelle. */
export function LiveCoverageSheet({ visible, onClose, summary, rows, onSelectDriver }: Props) {
  return (
    <EnterpriseBottomSheet
      visible={visible}
      onClose={onClose}
      title="Suivi en direct"
      subtitle={summary}
    >
      {rows.length === 0 ? (
        <AppText variant="caption" style={s.empty}>
          Aucun chauffeur à afficher
        </AppText>
      ) : (
        rows.map((row) => (
          <Pressable
            key={row.driverId}
            onPress={() => {
              onSelectDriver?.(row.driverId);
              onClose();
            }}
            disabled={!onSelectDriver}
            style={({ pressed }) => [s.row, pressed && onSelectDriver ? s.pressed : null]}
            accessibilityRole={onSelectDriver ? "button" : "text"}
            accessibilityLabel={
              row.lastPositionLabel
                ? `${row.initials}. ${row.name}. ${row.statusLabel}. ${row.lastPositionLabel}`
                : `${row.initials}. ${row.name}. ${row.statusLabel}`
            }
          >
            <View
              style={[
                s.dot,
                row.isLive ? s.dotLive : s.dotOffline,
              ]}
            />
            <AppText variant="caption" style={s.initials}>
              {row.initials}
            </AppText>
            <View style={s.rowText}>
              <AppText variant="body" style={s.name} numberOfLines={1}>
                {row.name}
              </AppText>
              {row.lastPositionLabel ? (
                <AppText variant="caption" style={s.lastSeen} numberOfLines={1}>
                  {row.lastPositionLabel}
                </AppText>
              ) : null}
            </View>
            <AppText variant="caption" style={[s.status, row.isLive ? s.statusLive : s.statusOffline]}>
              {row.statusLabel}
            </AppText>
          </Pressable>
        ))
      )}
    </EnterpriseBottomSheet>
  );
}

const s = StyleSheet.create({
  empty: {
    color: E.TEXT_SEC,
    paddingVertical: 8,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    minHeight: 44,
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderRadius: 10,
  },
  pressed: { opacity: 0.88 },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  dotLive: {
    backgroundColor: E.BRAND,
  },
  dotOffline: {
    backgroundColor: "transparent",
    borderWidth: 1.5,
    borderColor: E.TEXT_MUTED,
  },
  initials: {
    width: 28,
    color: E.TEXT,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  rowText: {
    flex: 1,
    minWidth: 0,
  },
  name: {
    color: E.TEXT,
    fontWeight: "600",
  },
  lastSeen: {
    color: E.TEXT_SEC,
    marginTop: 2,
  },
  status: {
    flexShrink: 0,
    fontSize: FONT_SIZE.px12,
    fontWeight: "600",
  },
  statusLive: { color: E.BRAND },
  statusOffline: { color: E.TEXT_SEC },
});
