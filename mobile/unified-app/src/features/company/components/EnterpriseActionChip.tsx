import type { ComponentProps, ReactNode } from "react";
import { ActivityIndicator, Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { E } from "../theme/enterpriseOpsTheme";

type IconName = ComponentProps<typeof Ionicons>["name"];

type EnterpriseActionChipProps = {
  icon: IconName;
  label: string;
  onPress: () => void;
  disabled?: boolean;
  /** Couleur de l'icône (défaut: brand) */
  iconColor?: string;
  /** Couleur du libellé (défaut: brand) */
  textColor?: string;
  showSpinner?: boolean;
  spinnerColor?: string;
  /** `muted` = neutre ; `details` = puce Détails (fond, bordure) comme l’appli de référence */
  tone?: "brand" | "muted" | "transfer" | "urgent" | "danger" | "urgentCta" | "details";
  /** Moins haut, pour barre prioritaire fixe */
  compact?: boolean;
  /** Icône légèrement plus visible */
  iconSize?: number;
};

const toneToColors = (tone: EnterpriseActionChipProps["tone"]) => {
  switch (tone) {
    case "muted":
      return { icon: E.TEXT_SEC, text: E.TEXT_SEC, bg: E.BG, border: E.BORDER };
    case "transfer":
      return { icon: E.TRANSFER, text: E.TRANSFER, bg: E.BG, border: E.BORDER };
    case "urgent":
      return { icon: E.URGENT, text: E.URGENT, bg: E.BG, border: E.BORDER };
    case "urgentCta":
      return { icon: "#FFFFFF", text: "#FFFFFF", bg: E.URGENT, border: "#D97706" };
    case "details":
      return { icon: E.TEXT_SEC, text: E.TEXT_SEC, bg: E.BG, border: E.BORDER };
    case "danger":
      return { icon: E.DANGER, text: E.DANGER, bg: E.BG, border: E.BORDER };
    case "brand":
    default:
      return { icon: E.BRAND, text: E.BRAND, bg: E.BG, border: E.BORDER };
  }
};

export function EnterpriseActionChip({
  icon,
  label,
  onPress,
  disabled,
  iconColor: iconColorOverride,
  textColor: textColorOverride,
  showSpinner,
  spinnerColor,
  tone = "brand",
  compact = false,
  iconSize: iconSizeOverride,
}: EnterpriseActionChipProps) {
  const t = toneToColors(tone);
  const iconColor = iconColorOverride ?? t.icon;
  const textColor = textColorOverride ?? t.text;
  const iconSize = iconSizeOverride ?? (tone === "urgentCta" || compact || tone === "details" ? 16 : 14);
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        s.chip,
        compact && s.chipCompact,
        tone === "details" && s.chipDetailsCase,
        { backgroundColor: t.bg, borderColor: t.border },
        tone === "urgentCta" && s.chipUrgentCtaShadow,
        (disabled || pressed) && s.chipDim,
        disabled && s.chipOff,
      ]}
    >
      {showSpinner ? (
        <ActivityIndicator size="small" color={spinnerColor ?? iconColor} />
      ) : (
        <Ionicons name={icon} size={iconSize} color={iconColor} />
      )}
      <Text
        style={[
          s.txt,
          compact && s.txtCompact,
          tone === "details" && s.txtDetailsCase,
          { color: textColor },
        ]}
        numberOfLines={1}
      >
        {label}
      </Text>
    </Pressable>
  );
}

/**
 * Remplace `footerActions` operations : rangée de puces qui wrap.
 */
export function EnterpriseFooterActionRow({ children }: { children: ReactNode }) {
  return <View style={s.row}>{children}</View>;
}

type PriorityRowProps = { children: ReactNode; /** Dans la même ligne que la pastille / chevron */ inline?: boolean };

/** Barre d’actions prioritaires (souvent sous l’en-tête) ; `inline` = entre la pastille et le chevron. */
export function EnterprisePriorityActionRow({ children, inline }: PriorityRowProps) {
  return <View style={inline ? s.priorityRowInline : s.priorityRow}>{children}</View>;
}

type RoundVariant = "urgent" | "assign";

/** Bouton rond, icône seule (réf. operations : urgence = fond ambre, assign = fond f7fc + bord, icône #00796B). */
type EnterpriseRoundIconActionProps = {
  icon: IconName;
  onPress: () => void;
  disabled?: boolean;
  accessibilityLabel: string;
  variant: RoundVariant;
  showSpinner?: boolean;
  spinnerColor?: string;
};

const roundUrgentShadow = Platform.select({
  web: { boxShadow: "0 1px 3px rgba(217,119,6,0.35)" } as const,
  default: {
    elevation: 2,
    shadowColor: E.URGENT,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.25,
    shadowRadius: 2,
  },
});

export function EnterpriseRoundIconAction({
  icon,
  onPress,
  disabled,
  accessibilityLabel,
  variant,
  showSpinner,
  spinnerColor,
}: EnterpriseRoundIconActionProps) {
  const urgent = variant === "urgent";
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      style={({ pressed }) => [
        s.roundBase,
        urgent ? s.roundUrgent : s.roundAssign,
        urgent && roundUrgentShadow,
        (pressed || disabled) && s.chipDim,
        disabled && s.chipOff,
      ]}
    >
      {showSpinner && urgent ? (
        <ActivityIndicator size="small" color={spinnerColor ?? "#FFFFFF"} />
      ) : (
        <Ionicons
          name={icon}
          size={urgent ? 16 : 15}
          color={urgent ? "#FFFFFF" : E.BRAND}
        />
      )}
    </Pressable>
  );
}

const s = StyleSheet.create({
  row: { flexDirection: "row", flexWrap: "wrap", gap: 6, marginTop: 0 },
  priorityRow: {
    flexDirection: "row" as const,
    flexWrap: "wrap" as const,
    alignItems: "center" as const,
    gap: 8,
    marginTop: 8,
    marginBottom: 2,
  },
  priorityRowInline: {
    flexDirection: "row" as const,
    flexWrap: "nowrap" as const,
    alignItems: "center" as const,
    gap: 4,
    marginTop: 0,
    marginBottom: 0,
  },
  roundBase: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  roundUrgent: {
    backgroundColor: E.URGENT,
    borderWidth: 1,
    borderColor: "#D97706",
  },
  roundAssign: {
    backgroundColor: E.BG,
    borderWidth: 1,
    borderColor: E.BORDER,
  },
  chip: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 4,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 8,
    backgroundColor: E.BG,
    borderWidth: 1,
  },
  chipCompact: { paddingVertical: 5, paddingHorizontal: 9, borderRadius: 8, gap: 3 },
  /** Puce « Détails » (operations / capture) */
  chipDetailsCase: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 8, gap: 6, alignItems: "center" as const },
  chipUrgentCtaShadow: { elevation: 2, shadowColor: E.URGENT, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.22, shadowRadius: 2 },
  chipDim: { opacity: 0.9 },
  chipOff: { opacity: 0.5 },
  txt: { fontSize: 12, fontWeight: "600" as const, flexShrink: 1 as const },
  txtCompact: { fontSize: 11, fontWeight: "700" as const },
  txtDetailsCase: { fontSize: 13, fontWeight: "600" as const, letterSpacing: 0.2 },
});
