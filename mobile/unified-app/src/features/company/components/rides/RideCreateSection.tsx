import type { ReactNode } from "react";
import { Pressable, StyleSheet, View, type ViewStyle } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";
import { E } from "../../theme/enterpriseOpsTheme";

export type RideCreateSectionBadgeTone = "required" | "recommended" | "success" | "muted";

export type RideCreateSectionBadge = {
  label: string;
  tone?: RideCreateSectionBadgeTone;
  /** Icône Ionicons optionnelle à gauche du label. */
  icon?: keyof typeof Ionicons.glyphMap;
};

export type RideCreateSectionProps = {
  number: number | string;
  title: string;
  /** Sous-titre fin sous le titre (optionnel). */
  subtitle?: string;
  /** Petit badge à droite (Requis, Recommandé, etc.). */
  badge?: RideCreateSectionBadge;
  /** Action à droite (ex. chevron pour accordéon, helper). */
  trailing?: ReactNode;
  /** Espace entre les enfants. */
  gap?: number;
  /** Style additionnel pour le conteneur de la carte. */
  style?: ViewStyle;
  /** Contenu masquable : on garde la card mais on cache le contenu. */
  hideBody?: boolean;
  /** La section est-elle complétée ? Si vrai, la pastille numéro devient un check vert. */
  complete?: boolean;
  /** Si fourni, le header devient pressable et toggle l'ouverture (chevron auto à droite). */
  onTogglePress?: () => void;
  /** Etat ouvert / fermé pour afficher le bon chevron. Si non fourni, déduit de `hideBody`. */
  open?: boolean;
  children?: ReactNode;
};

const TONE_STYLES: Record<RideCreateSectionBadgeTone, { bg: string; fg: string; border: string }> = {
  required: {
    bg: "rgba(0, 121, 107, 0.10)",
    fg: E.BRAND_DARK,
    border: "rgba(0, 121, 107, 0.28)",
  },
  recommended: {
    bg: "rgba(20, 184, 166, 0.12)",
    fg: "#0F766E",
    border: "rgba(20, 184, 166, 0.32)",
  },
  success: {
    bg: "rgba(0, 121, 107, 0.14)",
    fg: E.BRAND_DARK,
    border: E.BRAND,
  },
  muted: {
    bg: "rgba(148, 163, 184, 0.14)",
    fg: E.TEXT_SEC,
    border: "rgba(148, 163, 184, 0.32)",
  },
};

export function RideCreateSection({
  number,
  title,
  subtitle,
  badge,
  trailing,
  gap = 12,
  style,
  hideBody,
  complete,
  onTogglePress,
  open,
  children,
}: RideCreateSectionProps) {
  const badgeTone = badge ? TONE_STYLES[badge.tone ?? "required"] : null;
  const collapsible = typeof onTogglePress === "function";
  const isOpen = open ?? !hideBody;
  const autoChevron = collapsible ? (
    <Ionicons
      name={isOpen ? "chevron-up" : "chevron-down"}
      size={20}
      color={E.TEXT_MUTED}
    />
  ) : null;

  const headerContent = (
    <View style={s.header}>
      <View
        style={[s.numberPill, complete ? s.numberPillComplete : null]}
        accessibilityElementsHidden
        importantForAccessibility="no"
      >
        {complete ? (
          <Ionicons name="checkmark" size={14} color="#FFFFFF" />
        ) : (
          <AppText variant="label" style={s.numberPillText}>{String(number)}</AppText>
        )}
      </View>
      <View style={s.titleCol}>
        <AppText variant="sectionTitle" style={s.title}>{title}</AppText>
        {subtitle ? <AppText variant="caption" style={s.subtitle}>{subtitle}</AppText> : null}
      </View>
      {badge && badgeTone ? (
        <View
          style={[
            s.badge,
            { backgroundColor: badgeTone.bg, borderColor: badgeTone.border },
          ]}
        >
          {badge.icon ? (
            <Ionicons name={badge.icon} size={12} color={badgeTone.fg} />
          ) : null}
          <AppText variant="label" style={[s.badgeLabel, { color: badgeTone.fg }]}>{badge.label}</AppText>
        </View>
      ) : null}
      {trailing ?? autoChevron}
    </View>
  );

  return (
    <View style={[s.card, style]}>
      {collapsible ? (
        <Pressable
          onPress={onTogglePress}
          accessibilityRole="button"
          accessibilityState={{ expanded: isOpen }}
          accessibilityLabel={`${title}${isOpen ? ", ouvert" : ", fermé"}`}
          hitSlop={4}
        >
          {headerContent}
        </Pressable>
      ) : (
        headerContent
      )}
      {hideBody ? null : children != null ? (
        <View style={[s.body, { gap }]}>{children}</View>
      ) : null}
    </View>
  );
}

const s = StyleSheet.create({
  card: {
    paddingVertical: 2,
    gap: 8,
  },
  header: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 8,
    minHeight: 28,
  },
  numberPill: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: E.BRAND,
    alignItems: "center" as const,
    justifyContent: "center" as const,
  },
  numberPillComplete: {
    backgroundColor: E.BRAND_DARK,
  },
  numberPillText: {
    color: "#FFFFFF",
    fontWeight: "700" as const,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
  },
  titleCol: {
    flex: 1,
    minWidth: 0,
  },
  title: {
    color: E.TEXT,
    fontWeight: "700" as const,
    fontSize: FONT_SIZE.px14,
    lineHeight: 17,
  },
  subtitle: {
    color: E.TEXT_MUTED,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    marginTop: 1,
  },
  badge: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 4,
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: 999,
    borderWidth: 1,
  },
  badgeLabel: {
    fontSize: FONT_SIZE.px11,
    fontWeight: "700" as const,
    lineHeight: 14,
    letterSpacing: 0.2,
  },
  body: {
    gap: 12,
  },
});
