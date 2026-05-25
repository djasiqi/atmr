import type { PropsWithChildren, ReactNode } from "react";
import {
  Dimensions,
  Modal as NativeModal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { AppText } from "./AppText";
import { AppButton } from "./AppButton";
import { FONT_SIZE } from "../responsive/typographyTokens";

const SHEET_TEXT = "#1E293B";
const SHEET_TEXT_SEC = "#64748B";

export type LegacyModalPresentation = "centered" | "bottomSheet";

export type LegacyModalProps = PropsWithChildren<{
  visible: boolean;
  title: string;
  /** Sous-titre sous le titre (feuille réservation, etc.). */
  subtitle?: string;
  onClose: () => void;
  /** Présentation : carte centrée (défaut) ou feuille bas d’écran. */
  presentation?: LegacyModalPresentation;
  /** Remplace le bloc titre / sous-titre par un en-tête personnalisé. */
  renderHeader?: () => ReactNode;
  /**
   * Pied d’actions ; si défini, remplace le bouton « Fermer » par défaut.
   * Passez explicitement `null` pour supprimer entièrement la zone footer
   * (utile pour les bottom-sheets gérant leurs actions dans le corps).
   */
  footer?: ReactNode | null;
  /** Affiche la poignée (presentation bottomSheet). */
  showDragHandle?: boolean;
  /** Hauteur max du corps en mode bottomSheet (ratio de la hauteur écran). */
  sheetBodyMaxHeightRatio?: number;
}>;

/** Aligné sur les cartes dispatch (teinte marque légère). */
const MODAL_CARD_BORDER = "rgba(0, 121, 107, 0.1)";
const TITLE_COLOR = "#163A34";
const CTA_H = 44;
const CTA_R = 10;
const SECONDARY_CTA_BORDER = "rgba(148, 163, 184, 0.35)";
const MAX_BODY_H_CENTERED = Math.min(Dimensions.get("window").height * 0.62, 540);

const s = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(15, 23, 42, 0.4)",
    justifyContent: "center",
    padding: 18,
  },
  backdropSheet: {
    flex: 1,
    justifyContent: "flex-end",
    paddingHorizontal: 0,
    paddingBottom: 0,
    paddingTop: 0,
  },
  backdropDim: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(15, 23, 42, 0.4)",
  },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: MODAL_CARD_BORDER,
    paddingHorizontal: 18,
    paddingTop: 16,
    paddingBottom: 14,
    maxWidth: 520,
    width: "100%" as const,
    alignSelf: "center" as const,
    ...Platform.select({
      web: {
        boxShadow: "0 12px 40px rgba(15, 23, 42, 0.12)",
      },
      default: {
        elevation: 8,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.12,
        shadowRadius: 24,
      },
    }),
  },
  sheetCard: {
    backgroundColor: "#FFFFFF",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 0,
    maxWidth: "100%",
    width: "100%",
    paddingHorizontal: 18,
    paddingTop: 8,
    paddingBottom: 12,
    maxHeight: "88%",
    ...Platform.select({
      web: {
        boxShadow: "0 -8px 32px rgba(15, 23, 42, 0.14)",
      },
      default: {
        elevation: 16,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.12,
        shadowRadius: 20,
      },
    }),
  },
  dragHandle: {
    width: 44,
    height: 5,
    borderRadius: 999,
    backgroundColor: "rgba(148, 163, 184, 0.55)",
    alignSelf: "center",
    marginBottom: 12,
  },
  title: {
    color: TITLE_COLOR,
    fontSize: FONT_SIZE.px17,
    fontWeight: "700" as const,
    letterSpacing: 0.1,
    paddingBottom: 12,
    marginBottom: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.35)",
  },
  sheetTitleBlock: {
    paddingBottom: 4,
    marginBottom: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(148, 163, 184, 0.28)",
  },
  sheetTitle: {
    color: SHEET_TEXT,
    fontSize: FONT_SIZE.px18,
    fontWeight: "700" as const,
    letterSpacing: 0.15,
  },
  sheetSubtitle: {
    color: SHEET_TEXT_SEC,
    fontSize: FONT_SIZE.px13,
    fontWeight: "500" as const,
    marginTop: 6,
    lineHeight: 18,
  },
  body: {
    maxHeight: MAX_BODY_H_CENTERED,
  },
  bodySheet: {
    flexGrow: 0,
    flexShrink: 1,
    maxHeight: Dimensions.get("window").height * 0.62,
  },
  bodyContent: {
    paddingTop: 12,
    paddingBottom: 6,
  },
  bodyContentSheet: {
    paddingTop: 8,
    paddingBottom: 12,
  },
  footer: {
    marginTop: 8,
    paddingTop: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(148, 163, 184, 0.28)",
  },
});

/** Modal : carte centrée ou feuille bas d’écran, corps scrollable, pied d’actions optionnel. */
export function Modal({
  visible,
  title,
  subtitle,
  onClose,
  children,
  presentation = "centered",
  renderHeader,
  footer,
  showDragHandle = true,
  sheetBodyMaxHeightRatio = 0.62,
}: LegacyModalProps) {
  const insets = useSafeAreaInsets();
  const isSheet = presentation === "bottomSheet";

  const defaultFooter = (
    <AppButton
      title="Fermer"
      variant="secondary"
      onPress={onClose}
      style={{
        minHeight: CTA_H,
        borderRadius: CTA_R,
        borderColor: SECONDARY_CTA_BORDER,
      }}
    />
  );

  const headerNode =
    renderHeader?.() ??
    (isSheet ? (
      <View style={s.sheetTitleBlock}>
        <AppText variant="sectionTitle" style={s.sheetTitle} accessibilityRole="header">
          {title}
        </AppText>
        {subtitle ? (
          <AppText variant="bodyMuted" style={s.sheetSubtitle}>
            {subtitle}
          </AppText>
        ) : null}
      </View>
    ) : (
      <AppText variant="sectionTitle" style={s.title} accessibilityRole="header">
        {title}
      </AppText>
    ));

  const scrollBody = (
    <ScrollView
      style={isSheet ? [s.bodySheet, { maxHeight: Dimensions.get("window").height * sheetBodyMaxHeightRatio }] : s.body}
      contentContainerStyle={isSheet ? s.bodyContentSheet : s.bodyContent}
      keyboardShouldPersistTaps="handled"
      keyboardDismissMode={Platform.OS === "ios" ? "interactive" : "on-drag"}
      showsVerticalScrollIndicator={Platform.OS !== "web"}
    >
      {children}
    </ScrollView>
  );

  const footerNode =
    footer === null ? null : <View style={s.footer}>{footer ?? defaultFooter}</View>;

  if (isSheet) {
    return (
      <NativeModal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
        <View style={s.backdropSheet}>
          <Pressable style={s.backdropDim} onPress={onClose} accessibilityLabel="Fermer" />
          <View style={[s.sheetCard, { paddingBottom: Math.max(12, insets.bottom + 8) }]}>
            {showDragHandle ? <View style={s.dragHandle} /> : null}
            {headerNode}
            {scrollBody}
            {footerNode}
          </View>
        </View>
      </NativeModal>
    );
  }

  return (
    <NativeModal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={s.backdrop}>
        <View style={s.card}>
          {headerNode}
          {scrollBody}
          {footerNode}
        </View>
      </View>
    </NativeModal>
  );
}
