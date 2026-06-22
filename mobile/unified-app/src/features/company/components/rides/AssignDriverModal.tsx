import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton, AppSpinner, Modal } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { E } from "../../theme/enterpriseOpsTheme";

export type AssignDriverOption = { id: number; label: string };

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

  return (
    <Modal
      visible={visible}
      title={title}
      onClose={() => {
        if (!pending) onClose();
      }}
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
      {!pending && drivers.length === 0 ? (
        <AppText variant="bodyMuted" style={styles.emptyText}>
          Aucun chauffeur disponible pour cette course.
        </AppText>
      ) : null}
      {drivers.map((driver) => {
        const selected = selectedDriverId === driver.id;
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
            accessibilityLabel={driver.label}
          >
            <AppText
              variant="body"
              style={[styles.rowLabel, selected ? styles.rowLabelSelected : null]}
              numberOfLines={2}
            >
              {driver.label}
            </AppText>
            {selected ? (
              <Ionicons name="checkmark-circle" size={20} color={E.BRAND} accessibilityElementsHidden />
            ) : null}
          </Pressable>
        );
      })}
    </Modal>
  );
}

const styles = StyleSheet.create({
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
    justifyContent: "space-between",
    gap: 12,
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 16,
    marginBottom: 8,
    minHeight: 52,
  },
  rowNormal: {
    borderColor: E.BORDER,
    backgroundColor: E.CARD,
  },
  rowSelected: {
    borderColor: E.BRAND,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  rowPressed: {
    opacity: 0.92,
  },
  rowLabel: {
    flex: 1,
    color: E.TEXT,
    fontWeight: "600",
    fontSize: FONT_SIZE.px16,
    lineHeight: 20,
  },
  rowLabelSelected: {
    color: E.BRAND_DARK,
    fontWeight: "700",
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
