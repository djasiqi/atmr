import type { ReactNode } from "react";
import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { E } from "../theme/enterpriseOpsTheme";

export type EnterpriseBottomSheetProps = {
  visible: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: ReactNode;
  /** Contenu scrollable (listes d’options). */
  scrollable?: boolean;
};

/**
 * Feuille modale bas d’écran (animation slide), alignée sur le pattern `CompanyInboxButton`.
 */
export function EnterpriseBottomSheet({
  visible,
  onClose,
  title,
  subtitle,
  children,
  scrollable = true,
}: EnterpriseBottomSheetProps) {
  const insets = useSafeAreaInsets();
  const body = scrollable ? (
    <ScrollView
      style={s.scroll}
      contentContainerStyle={s.scrollContent}
      keyboardShouldPersistTaps="handled"
      showsVerticalScrollIndicator
    >
      {children}
    </ScrollView>
  ) : (
    <View style={s.staticBody}>{children}</View>
  );

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={s.backdrop}>
        <Pressable style={s.backdropTap} onPress={onClose} accessibilityLabel="Fermer" />
        <View style={[s.card, { paddingBottom: Math.max(20, insets.bottom + 12) }]}>
          <View style={s.header}>
            <Text style={s.title}>{title}</Text>
            {subtitle ? <Text style={s.subtitle}>{subtitle}</Text> : null}
          </View>
          {body}
          <Pressable
            onPress={onClose}
            accessibilityRole="button"
            accessibilityLabel="Fermer"
            style={({ pressed }) => [s.closeCta, pressed && { opacity: 0.88 }]}
          >
            <Text style={s.closeCtaText}>Fermer</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

const s = StyleSheet.create({
  backdrop: { flex: 1, justifyContent: "flex-end" },
  backdropTap: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(0,0,0,0.4)" },
  card: {
    backgroundColor: E.CARD,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    paddingTop: 14,
    paddingHorizontal: 16,
    paddingBottom: 20,
    maxHeight: "88%",
  },
  header: {
    marginBottom: 16,
  },
  /** Réf. maquette : 18px / 22 lh, semibold. */
  title: {
    color: E.TEXT,
    fontSize: 18,
    lineHeight: 22,
    fontWeight: "600" as const,
  },
  subtitle: {
    color: E.TEXT_SEC,
    fontSize: 13,
    lineHeight: 16,
    fontWeight: "600" as const,
    marginTop: 8,
  },
  scroll: { maxHeight: 360 },
  scrollContent: { paddingBottom: 12, gap: 8 },
  staticBody: { gap: 8, marginBottom: 8 },
  closeCta: {
    marginTop: 16,
    minHeight: 48,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.45)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  closeCtaText: {
    color: "#163A34",
    fontSize: 13,
    lineHeight: 16,
    fontWeight: "600" as const,
  },
});
