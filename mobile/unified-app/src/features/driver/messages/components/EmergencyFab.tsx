import { Modal, Pressable, StyleSheet, View } from "react-native";
import { useState } from "react";
import { AppText } from "../../../../design/ui/AppText";
import type { EmergencyIssueType } from "../types";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

const ISSUES: { type: EmergencyIssueType; label: string }[] = [
  { type: "patient_absent", label: "Patient absent" },
  { type: "retard_important", label: "Retard important" },
  { type: "panne_vehicule", label: "Panne véhicule" },
  { type: "incident", label: "Incident" },
  { type: "besoin_assistance", label: "Besoin assistance" },
];

type Props = {
  onReport: (issue: EmergencyIssueType) => void;
  pending?: boolean;
  /** `fab` = flottant sur le fil ; `header` = pillule dans la barre du haut. */
  variant?: "fab" | "header";
};

export function EmergencyFab({ onReport, pending, variant = "fab" }: Props) {
  const [open, setOpen] = useState(false);
  const isHeader = variant === "header";

  return (
    <>
      <Pressable
        style={[
          isHeader ? styles.headerBtn : styles.fab,
          pending && styles.pending,
        ]}
        onPress={() => setOpen(true)}
        accessibilityRole="button"
        accessibilityLabel="Signaler un problème"
      >
        <AppText variant="label" style={styles.btnText}>
          ⚠ Problème
        </AppText>
      </Pressable>
      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <Pressable style={styles.backdrop} onPress={() => setOpen(false)}>
          <View style={styles.sheet}>
            <AppText variant="sectionTitle" style={styles.title}>
              Signaler un problème
            </AppText>
            {ISSUES.map((item) => (
              <Pressable
                key={item.type}
                style={styles.option}
                onPress={() => {
                  setOpen(false);
                  onReport(item.type);
                }}
              >
                <AppText variant="body">{item.label}</AppText>
              </Pressable>
            ))}
          </View>
        </Pressable>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  fab: {
    position: "absolute",
    right: 16,
    bottom: 88,
    backgroundColor: "#b91c1c",
    borderRadius: 24,
    paddingHorizontal: 14,
    paddingVertical: 10,
    elevation: 4,
    zIndex: 20,
  },  pending: { opacity: 0.7 },
  btnText: { color: "#fff", fontWeight: "700", fontSize: FONT_SIZE.px12 },
  headerBtn: {
    backgroundColor: "#b91c1c",
    borderRadius: 16,
    paddingHorizontal: 10,
    paddingVertical: 6,
    marginLeft: 4,
  },
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.45)",
    justifyContent: "flex-end",
  },
  sheet: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 16,
    gap: 4,
  },
  title: { marginBottom: 8 },
  option: {
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#e5e7eb",
  },
});
