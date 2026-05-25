import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { AppModal } from "../../../design/ui/AppModal";
import { M } from "../messagingTheme";

export type ConfirmDialogModalProps = {
  visible: boolean;
  title: string;
  icon?: keyof typeof Ionicons.glyphMap;
  iconColor?: string;
  iconBackground?: string;
  message?: string;
  children?: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  destructive?: boolean;
  pending?: boolean;
  onClose: () => void;
  onConfirm: () => void;
};

export function ConfirmDialogModal({
  visible,
  title,
  icon,
  iconColor = M.DANGER,
  iconBackground = M.DANGER_SOFT,
  message,
  children,
  confirmLabel,
  cancelLabel = "Annuler",
  destructive = false,
  pending = false,
  onClose,
  onConfirm,
}: ConfirmDialogModalProps) {
  return (
    <AppModal
      visible={visible}
      onClose={onClose}
      variant="dialog"
      backdropColor="#0f172a"
      backdropOpacity={0.5}
      containerStyle={styles.cardWrap}
      screen="messages.confirm_dialog"
    >
      <Pressable style={styles.card} onPress={(e) => e.stopPropagation()}>
        {icon ? (
          <View style={[styles.iconWrap, { backgroundColor: iconBackground }]}>
            <Ionicons name={icon} size={28} color={iconColor} />
          </View>
        ) : null}
        <AppText variant="sectionTitle" style={styles.title}>
          {title}
        </AppText>
        {children}
        {message ? (
          <AppText variant="bodyMuted" style={styles.message}>
            {message}
          </AppText>
        ) : null}
        <View style={styles.actions}>
          <Pressable
            style={styles.btnSecondary}
            onPress={onClose}
            disabled={pending}
            accessibilityLabel={cancelLabel}
          >
            <AppText variant="label" style={styles.btnSecondaryText}>
              {cancelLabel}
            </AppText>
          </Pressable>
          <Pressable
            style={[
              destructive ? styles.btnDanger : styles.btnPrimary,
              pending && styles.btnDisabled,
            ]}
            onPress={onConfirm}
            disabled={pending}
            accessibilityLabel={confirmLabel}
          >
            {pending ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <AppText variant="label" style={styles.btnPrimaryText}>
                {confirmLabel}
              </AppText>
            )}
          </Pressable>
        </View>
      </Pressable>
    </AppModal>
  );
}

const styles = StyleSheet.create({
  cardWrap: {
    left: 28,
    right: 28,
    top: "30%",
    alignItems: "center",
  },
  card: {
    width: "100%",
    maxWidth: 340,
    backgroundColor: M.CARD,
    borderRadius: 18,
    paddingHorizontal: 22,
    paddingTop: 22,
    paddingBottom: 18,
    gap: 12,
    shadowColor: "#0f172a",
    shadowOpacity: 0.12,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
  },
  iconWrap: {
    alignSelf: "center",
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: "center",
    justifyContent: "center",
  },
  title: { textAlign: "center", color: M.TEXT },
  message: { textAlign: "center", lineHeight: 21 },
  actions: { flexDirection: "row", gap: 10, marginTop: 4 },
  btnSecondary: {
    flex: 1,
    paddingVertical: 13,
    borderRadius: 12,
    backgroundColor: "#f1f5f9",
    alignItems: "center",
  },
  btnSecondaryText: { color: M.TEXT, fontWeight: "600" },
  btnPrimary: {
    flex: 1,
    paddingVertical: 13,
    borderRadius: 12,
    backgroundColor: M.BRAND,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 46,
  },
  btnDanger: {
    flex: 1,
    paddingVertical: 13,
    borderRadius: 12,
    backgroundColor: M.DANGER,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 46,
  },
  btnPrimaryText: { color: "#fff", fontWeight: "700" },
  btnDisabled: { opacity: 0.55 },
});
