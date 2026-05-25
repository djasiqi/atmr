import type { ReactNode } from "react";
import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useBottomSheetLayout } from "../../../design/responsive";
import { E } from "../theme/enterpriseOpsTheme";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

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
  const sheet = useBottomSheetLayout({ reservedChromeHeight: 132 });
  const body = scrollable ? (
    <ScrollView
      style={[s.scroll, { maxHeight: sheet.scrollMaxHeight }]}
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
        <View style={[s.card, { maxHeight: sheet.cardMaxHeight, paddingBottom: sheet.paddingBottom }]}>
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
  },
  header: {
    marginBottom: 16,
  },
  /** Réf. maquette : 18px / 22 lh, semibold. */
  title: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px18,
    lineHeight: 22,
    fontWeight: "600" as const,
  },
  subtitle: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600" as const,
    marginTop: 8,
  },
  scroll: {},
  scrollContent: { paddingBottom: 12, gap: 8 },
  staticBody: { gap: 8, marginBottom: 8 },
  closeCta: {
    marginTop: 16,
    minHeight: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(148, 163, 184, 0.35)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  closeCtaText: {
    color: "#334155",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "500" as const,
  },
});
