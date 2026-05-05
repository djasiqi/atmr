import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { EnterpriseBottomSheet } from "./EnterpriseBottomSheet";
import { E } from "../theme/enterpriseOpsTheme";

export type DispatchModeValue = "manual" | "semi_auto" | "fully_auto";

export type DispatchModeSheetProps = {
  visible: boolean;
  mode: DispatchModeValue | null;
  onClose: () => void;
  /** Si défini, une ligne déclenche le changement côté API puis le parent ferme la feuille. */
  onSelectMode?: (mode: DispatchModeValue) => void | Promise<void>;
  /** Permission `company:dispatch:manage` (ou équivalent produit). */
  switchingEnabled?: boolean;
};

type RowDef = {
  value: DispatchModeValue;
  title: string;
  subtitle: string;
};

const ROWS: RowDef[] = [
  { value: "manual", title: "Manuel", subtitle: "Dispatch à la main" },
  { value: "semi_auto", title: "Semi-auto", subtitle: "Suggestions et validation" },
  { value: "fully_auto", title: "Auto", subtitle: "Automatisation maximale" },
];

export function DispatchModeSheet({
  visible,
  mode,
  onClose,
  onSelectMode,
  switchingEnabled = true,
}: DispatchModeSheetProps) {
  const canSwitch = Boolean(onSelectMode) && switchingEnabled;

  return (
    <EnterpriseBottomSheet
      visible={visible}
      onClose={onClose}
      title="Mode de dispatch"
      subtitle="Choisissez votre mode d’assignation"
      scrollable={false}
    >
      <View style={s.list}>
        {ROWS.map((row) => {
          const selected =
            mode === row.value || (row.value === "manual" && (mode === null || mode === undefined));
          const inactive = !canSwitch;
          return (
            <Pressable
              key={row.value}
              disabled={inactive}
              onPress={() => {
                if (inactive || !onSelectMode) return;
                void onSelectMode(row.value);
              }}
              style={({ pressed }) => [
                s.row,
                selected && !inactive ? s.rowSelected : s.rowPlain,
                inactive && s.rowDisabled,
                pressed && !inactive && s.pressed,
              ]}
              accessibilityRole="button"
              accessibilityLabel={`${row.title}. ${row.subtitle}${selected ? " — sélectionné" : ""}`}
              accessibilityState={{ selected, disabled: inactive }}
            >
              <View style={s.rowLeft}>
                <View style={s.titles}>
                  <AppText
                    variant="label"
                    style={[s.title, inactive && s.muted, selected && !inactive && s.titleSelected]}
                  >
                    {row.title}
                  </AppText>
                  <AppText variant="caption" style={[s.sub, inactive && s.muted]}>
                    {row.subtitle}
                  </AppText>
                </View>
              </View>
              {selected ? (
                <Ionicons
                  name="checkmark-circle"
                  size={22}
                  color={inactive ? E.TEXT_MUTED : E.BRAND}
                  accessibilityElementsHidden
                />
              ) : (
                <View style={s.checkPlaceholder} />
              )}
            </Pressable>
          );
        })}
      </View>
      {!switchingEnabled ? (
        <AppText variant="caption" style={s.denied}>
          Vous n’avez pas la permission de changer le mode ici.
        </AppText>
      ) : null}
    </EnterpriseBottomSheet>
  );
}

const s = StyleSheet.create({
  list: { gap: 10 },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
    paddingVertical: 14,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
  },
  rowPlain: {
    borderColor: E.BORDER,
    backgroundColor: E.CARD,
  },
  rowSelected: {
    borderColor: E.BRAND,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  rowDisabled: { opacity: 0.45 },
  pressed: { opacity: 0.92 },
  rowLeft: { flex: 1, minWidth: 0 },
  titles: { gap: 2 },
  title: { color: E.TEXT, fontWeight: "700" as const },
  titleSelected: { color: E.BRAND },
  sub: { color: E.TEXT_SEC, fontWeight: "500" as const },
  muted: { color: E.TEXT_SEC },
  denied: { color: E.TEXT_SEC, marginTop: 12 },
  checkPlaceholder: { width: 22, height: 22 },
});
