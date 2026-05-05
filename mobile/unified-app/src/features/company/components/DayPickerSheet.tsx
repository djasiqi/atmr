import { Platform, Pressable, StyleSheet, Text, View } from "react-native";
import dayjs from "dayjs";
import "dayjs/locale/fr";
import { EnterpriseBottomSheet } from "./EnterpriseBottomSheet";
import { E } from "../theme/enterpriseOpsTheme";

dayjs.locale("fr");

const ROW_BORDER = "rgba(145, 165, 157, 0.38)";
const BADGE_COLOR = E.BRAND;

function nextSevenIsoDates(): string[] {
  const start = dayjs().startOf("day");
  return Array.from({ length: 7 }, (_, i) => start.add(i, "day").format("YYYY-MM-DD"));
}

/** Ligne du jour : badge « Aujourd'hui » / « Demain » + date complète, ou une seule ligne « Jeudi 7 mai ». */
function formatDayRows(iso: string): { badge: string | null; line: string } {
  const d = dayjs(iso);
  const today = dayjs().startOf("day");
  const tomorrow = today.add(1, "day");
  const lineRaw = d.format("dddd D MMMM");
  const line = lineRaw.charAt(0).toUpperCase() + lineRaw.slice(1);
  if (d.isSame(today, "day")) return { badge: "Aujourd'hui", line };
  if (d.isSame(tomorrow, "day")) return { badge: "Demain", line };
  return { badge: null, line };
}

export type DayPickerSheetProps = {
  visible: boolean;
  selectedDate: string;
  onClose: () => void;
  onSelectDate: (isoDate: string) => void;
};

export function DayPickerSheet({ visible, selectedDate, onClose, onSelectDate }: DayPickerSheetProps) {
  const days = nextSevenIsoDates();

  return (
    <EnterpriseBottomSheet visible={visible} onClose={onClose} title="Choisir une date" subtitle="Courses des 7 prochains jours">
      {days.map((iso) => {
        const { badge, line } = formatDayRows(iso);
        return (
          <Pressable
            key={iso}
            onPress={() => onSelectDate(iso)}
            style={({ pressed }) => [s.row, pressed && s.pressed]}
            accessibilityRole="button"
            accessibilityState={{ selected: iso === selectedDate }}
            accessibilityLabel={badge ? `${badge}, ${line}` : line}
          >
            <View style={s.rowMain}>
              {badge ? (
                <View style={s.twoLines}>
                  <Text style={s.badge} {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}>
                    {badge}
                  </Text>
                  <Text
                    style={s.datePrimary}
                    numberOfLines={2}
                    {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
                  >
                    {line}
                  </Text>
                </View>
              ) : (
                <Text
                  style={s.datePrimary}
                  numberOfLines={2}
                  {...(Platform.OS === "android" ? { includeFontPadding: false } : {})}
                >
                  {line}
                </Text>
              )}
            </View>
            <View style={s.trailSpacer} />
          </Pressable>
        );
      })}
    </EnterpriseBottomSheet>
  );
}

const s = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    paddingVertical: 14,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: ROW_BORDER,
    backgroundColor: E.CARD,
  },
  pressed: { opacity: 0.92 },
  rowMain: {
    flex: 1,
    minWidth: 0,
  },
  twoLines: {
    gap: 4,
  },
  /** Réf. « Aujourd'hui » / « Demain » — 13px, accent marque. */
  badge: {
    fontSize: 13,
    lineHeight: 16,
    fontWeight: "600" as const,
    color: BADGE_COLOR,
  },
  /** Réf. ligne principale — 16px / 20 lh (vw87k0 / 1djweci). */
  datePrimary: {
    fontSize: 16,
    lineHeight: 20,
    fontWeight: "600" as const,
    color: E.TEXT,
  },
  /** Colonne vide pour aligner comme la maquette (pas de coche). */
  trailSpacer: {
    width: 40,
    height: 40,
  },
});
