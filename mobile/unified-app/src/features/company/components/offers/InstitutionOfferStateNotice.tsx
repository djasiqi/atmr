import { StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppButton } from "../../../../design/responsive";
import { AppText } from "../../../../design/ui/AppText";
import { E } from "../../theme/enterpriseOpsTheme";

export type InstitutionOfferStateNoticeVariant =
  | "expired"
  | "unavailable"
  | "rejected"
  | "cancelled"
  | "converted";

type InstitutionOfferStateNoticeProps = {
  variant: InstitutionOfferStateNoticeVariant;
  expiresLabel?: string | null;
  onPrimaryPress: () => void;
  primaryLabel?: string;
};

const COPY: Record<
  InstitutionOfferStateNoticeVariant,
  {
    icon: keyof typeof Ionicons.glyphMap;
    title: string;
    message: string;
    defaultPrimaryLabel: string;
    tone: "neutral" | "warning";
  }
> = {
  expired: {
    icon: "time-outline",
    title: "Offre expirée",
    message:
      "Le délai de réponse est écoulé. Cette demande n'est plus disponible pour acceptation ou proposition d'horaire.",
    defaultPrimaryLabel: "Retour aux demandes",
    tone: "neutral",
  },
  unavailable: {
    icon: "checkmark-done-outline",
    title: "Demande attribuée",
    message:
      "Un autre transporteur a accepté cette demande. Aucune action n'est possible de votre côté.",
    defaultPrimaryLabel: "Retour aux demandes",
    tone: "neutral",
  },
  rejected: {
    icon: "close-circle-outline",
    title: "Offre refusée",
    message: "Vous avez déjà refusé cette demande institution.",
    defaultPrimaryLabel: "Retour aux demandes",
    tone: "neutral",
  },
  cancelled: {
    icon: "ban-outline",
    title: "Demande annulée",
    message: "L'institution a annulé cette demande de transport.",
    defaultPrimaryLabel: "Retour aux demandes",
    tone: "warning",
  },
  converted: {
    icon: "swap-horizontal-outline",
    title: "Demande convertie",
    message: "Cette demande a déjà été convertie en course active.",
    defaultPrimaryLabel: "Retour aux demandes",
    tone: "neutral",
  },
};

export function InstitutionOfferStateNotice({
  variant,
  expiresLabel,
  onPrimaryPress,
  primaryLabel,
}: InstitutionOfferStateNoticeProps) {
  const config = COPY[variant];
  const isWarning = config.tone === "warning";

  return (
    <View
      style={[
        s.card,
        isWarning ? s.cardWarning : s.cardNeutral,
      ]}
    >
      <View style={s.headerRow}>
        <View style={[s.iconPill, isWarning ? s.iconPillWarning : s.iconPillNeutral]}>
          <Ionicons
            name={config.icon}
            size={20}
            color={isWarning ? E.URGENT : E.TEXT_SEC}
          />
        </View>
        <View style={s.headerCopy}>
          <AppText variant="label" style={s.title}>
            {config.title}
          </AppText>
          <AppText variant="bodyMuted" style={s.message}>
            {config.message}
          </AppText>
          {variant === "expired" && expiresLabel ? (
            <AppText variant="caption" style={s.meta}>
              {`Expiration : ${expiresLabel}`}
            </AppText>
          ) : null}
        </View>
      </View>
      <AppButton
        title={primaryLabel ?? config.defaultPrimaryLabel}
        variant="secondary"
        onPress={onPrimaryPress}
        style={s.primaryBtn}
      />
    </View>
  );
}

const s = StyleSheet.create({
  card: {
    borderRadius: 12,
    borderWidth: 1,
    padding: 14,
    marginBottom: 12,
    gap: 14,
  },
  cardNeutral: {
    backgroundColor: "#F8FAFC",
    borderColor: E.SHELL_BORDER,
  },
  cardWarning: {
    backgroundColor: "rgba(245, 158, 11, 0.08)",
    borderColor: "rgba(245, 158, 11, 0.25)",
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
  },
  iconPill: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  iconPillNeutral: {
    backgroundColor: "rgba(100, 116, 139, 0.12)",
  },
  iconPillWarning: {
    backgroundColor: "rgba(245, 158, 11, 0.15)",
  },
  headerCopy: {
    flex: 1,
    gap: 4,
    minWidth: 0,
  },
  title: {
    color: E.TEXT,
    fontSize: 15,
    lineHeight: 20,
  },
  message: {
    color: E.TEXT_SEC,
    fontSize: 14,
    lineHeight: 20,
  },
  meta: {
    color: E.TEXT_MUTED,
    marginTop: 2,
    fontSize: 12,
    lineHeight: 16,
  },
  primaryBtn: {
    alignSelf: "stretch",
  },
});
