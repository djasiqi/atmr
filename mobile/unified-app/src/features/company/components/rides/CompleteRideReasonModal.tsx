import { StyleSheet, TextInput, View } from "react-native";
import { AppButton, Modal } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";

type CompleteRideReasonModalProps = {
  visible: boolean;
  pending?: boolean;
  reason: string;
  error?: string | null;
  onChangeReason: (value: string) => void;
  onConfirm: () => void;
  onClose: () => void;
};

export function CompleteRideReasonModal({
  visible,
  pending = false,
  reason,
  error,
  onChangeReason,
  onConfirm,
  onClose,
}: CompleteRideReasonModalProps) {
  const trimmed = reason.trim();
  return (
    <Modal
      visible={visible}
      title="Valider la course"
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
              title="Annuler"
              variant="secondary"
              onPress={onClose}
              disabled={pending}
              style={styles.footerBtn}
            />
            <AppButton
              title={pending ? "Validation…" : "Valider"}
              variant="primary"
              onPress={onConfirm}
              disabled={pending || trimmed.length === 0}
              style={styles.footerBtn}
            />
          </View>
        </View>
      }
    >
      <AppText variant="bodyMuted" style={styles.hint}>
        Un motif est requis pour clôturer une course en route.
      </AppText>
      <TextInput
        value={reason}
        onChangeText={onChangeReason}
        editable={!pending}
        placeholder="Motif"
        placeholderTextColor={E.TEXT_SEC}
        style={styles.input}
        accessibilityLabel="Motif de clôture"
      />
    </Modal>
  );
}

const styles = StyleSheet.create({
  hint: {
    marginBottom: 10,
    lineHeight: 20,
  },
  input: {
    minHeight: 44,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.45)",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: E.TEXT,
    backgroundColor: E.BG,
  },
  footerWrap: {
    gap: 8,
  },
  footerRow: {
    flexDirection: "row",
    gap: 8,
  },
  footerBtn: {
    flex: 1,
    minHeight: 44,
    borderRadius: 10,
  },
  errorText: {
    color: E.DANGER,
    lineHeight: 19,
  },
});
