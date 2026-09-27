import { useEffect, useMemo, useState } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, AppSpinner, Modal } from "../../../../design/responsive";
import { AppInput } from "../../../../design/ui/AppInput";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  driverAvatarTone,
  driverPickerSummary,
  filterDriverOptions,
  uniqueDriverInitials,
  type AssignDriverPickerOption,
} from "./assignDriverPicker";

export type AssignDriverOption = AssignDriverPickerOption;

type AssignDriverModalProps = {
  visible: boolean;
  pending?: boolean;
  drivers: AssignDriverOption[];
  selectedDriverId: number | null;
  error?: string | null;
  onSelect: (id: number) => void;
  onConfirm: () => void;
  onClose: () => void;
  mode?: "assign" | "reassign";
};

export function AssignDriverModal({
  visible,
  pending = false,
  drivers,
  selectedDriverId,
  error,
  onSelect,
  onConfirm,
  onClose,
  mode = "assign",
}: AssignDriverModalProps) {
  const title = mode === "reassign" ? "Réassigner un chauffeur" : "Assigner un chauffeur";
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!visible) setQuery("");
  }, [visible]);

  const visibleDrivers = useMemo(() => filterDriverOptions(drivers, query), [drivers, query]);
  const initialsById = useMemo(() => uniqueDriverInitials(visibleDrivers), [visibleDrivers]);
  const summary = driverPickerSummary(drivers.length, visibleDrivers.length, query);
  const showSearch = drivers.length > 1;

  return (
    <Modal
      visible={visible}
      title={title}
      subtitle={summary}
      presentation="bottomSheet"
      reservedChromeHeight={showSearch ? 168 : 88}
      onClose={() => {
        if (!pending) onClose();
      }}
      renderHeader={
        showSearch
          ? () => (
              <View style={styles.header}>
                <AppText variant="sectionTitle" style={styles.headerTitle} accessibilityRole="header">
                  {title}
                </AppText>
                <AppText variant="caption" style={styles.headerSummary}>
                  {summary}
                </AppText>
                <AppInput
                  value={query}
                  onChangeText={setQuery}
                  placeholder="Rechercher un chauffeur"
                  autoCorrect={false}
                  autoCapitalize="none"
                  leftSlot={<Ionicons name="search-outline" size={18} color={E.TEXT_SEC} />}
                  shellStyle={styles.searchShell}
                  accessibilityLabel="Rechercher un chauffeur"
                />
              </View>
            )
          : undefined
      }
      footer={
        <View style={styles.footerWrap}>
          {error ? (
            <AppText variant="error" style={styles.errorText} accessibilityRole="alert">
              {error}
            </AppText>
          ) : null}
          <View style={styles.footerRow}>
            <AppButton
              title="Fermer"
              variant="secondary"
              onPress={onClose}
              disabled={pending}
              style={styles.footerBtnSecondary}
            />
            <AppButton
              title={pending ? "Assignation…" : "Confirmer"}
              variant="primary"
              onPress={onConfirm}
              disabled={pending || selectedDriverId == null || drivers.length === 0}
              style={styles.footerBtnPrimary}
            />
          </View>
        </View>
      }
    >
      {pending && drivers.length === 0 ? (
        <View style={styles.spinnerWrap}>
          <AppSpinner />
        </View>
      ) : null}
      {!pending && drivers.length === 0 && !error ? (
        <AppText variant="bodyMuted" style={styles.emptyText}>
          Aucun chauffeur disponible pour cette course.
        </AppText>
      ) : null}
      {!pending && drivers.length > 0 && visibleDrivers.length === 0 ? (
        <AppText variant="bodyMuted" style={styles.emptyText}>
          Aucun chauffeur ne correspond à cette recherche.
        </AppText>
      ) : null}
      {visibleDrivers.map((driver) => {
        const selected = selectedDriverId === driver.id;
        const tone = driverAvatarTone(driver.label);
        return (
          <Pressable
            key={driver.id}
            onPress={() => onSelect(driver.id)}
            disabled={pending}
            style={({ pressed }) => [
              styles.row,
              selected ? styles.rowSelected : styles.rowNormal,
              pressed && !pending ? styles.rowPressed : null,
            ]}
            accessibilityRole="button"
            accessibilityState={{ selected }}
            accessibilityLabel={selected ? `${driver.label}, sélectionné` : driver.label}
          >
            <View style={[styles.avatar, { backgroundColor: tone.background }]}>
              <AppText variant="label" style={[styles.avatarText, { color: tone.foreground }]}>
                {initialsById.get(driver.id) ?? "?"}
              </AppText>
            </View>
            <View style={styles.rowText}>
              <AppText
                variant="body"
                style={[styles.rowLabel, selected ? styles.rowLabelSelected : null]}
                numberOfLines={1}
              >
                {driver.label}
              </AppText>
              <AppText variant="caption" style={styles.rowHint} numberOfLines={1}>
                Disponible
              </AppText>
            </View>
            <View style={[styles.radio, selected ? styles.radioOn : null]}>
              {selected ? (
                <Ionicons name="checkmark" size={14} color="#FFFFFF" accessibilityElementsHidden />
              ) : null}
            </View>
          </Pressable>
        );
      })}
    </Modal>
  );
}

const styles = StyleSheet.create({
  header: {
    paddingBottom: 8,
    marginBottom: 4,
    gap: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.28)",
  },
  headerTitle: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px18,
    fontWeight: "700",
    letterSpacing: 0.15,
  },
  headerSummary: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px13,
    fontWeight: "500",
  },
  searchShell: {
    borderRadius: 12,
    minHeight: 44,
    paddingHorizontal: 10,
  },
  spinnerWrap: {
    paddingVertical: 16,
    alignItems: "center",
  },
  emptyText: {
    color: E.TEXT_SEC,
    marginBottom: 8,
    lineHeight: 20,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    borderWidth: 1,
    borderRadius: 16,
    paddingVertical: 10,
    paddingHorizontal: 12,
    marginBottom: 8,
    minHeight: 64,
  },
  rowNormal: {
    borderColor: "#E2E8F0",
    backgroundColor: E.CARD,
  },
  rowSelected: {
    borderColor: E.BRAND,
    backgroundColor: "#F0FDFA",
  },
  rowPressed: {
    opacity: 0.92,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: {
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  rowText: {
    flex: 1,
    gap: 2,
  },
  rowLabel: {
    color: E.TEXT,
    fontWeight: "600",
    fontSize: FONT_SIZE.px16,
    lineHeight: 20,
  },
  rowLabelSelected: {
    color: E.BRAND_DARK,
    fontWeight: "700",
  },
  rowHint: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
  },
  radio: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1.5,
    borderColor: "#CBD5E1",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  radioOn: {
    borderColor: E.BRAND,
    backgroundColor: E.BRAND,
  },
  footerWrap: {
    gap: 8,
  },
  footerRow: {
    flexDirection: "row",
    gap: 8,
  },
  footerBtnSecondary: {
    flex: 1,
    minHeight: 44,
    borderRadius: 10,
    borderColor: "rgba(148, 163, 184, 0.35)",
  },
  footerBtnPrimary: {
    flex: 1,
    minHeight: 44,
    borderRadius: 12,
  },
  errorText: {
    color: E.DANGER,
    lineHeight: 19,
  },
});
