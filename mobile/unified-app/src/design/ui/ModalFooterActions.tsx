import type { ReactNode } from "react";
import { StyleSheet, View } from "react-native";
import { useAccessibilityScale } from "../responsive/useAccessibilityScale";
import { useResponsiveTokens } from "../responsive/useResponsiveTokens";

export type ModalFooterActionsProps = {
  /** Bouton secondaire (gauche / haut). */
  secondary?: ReactNode;
  /** Bouton primaire (droite / bas). */
  primary: ReactNode;
  /** Contenu au-dessus des boutons (hint, etc.). */
  hint?: ReactNode;
  /** Forcer la disposition ; défaut = shouldStackRows. */
  stacked?: boolean;
};

/**
 * Pied d’actions de modale : row → column selon `shouldStackRows`.
 * Les enfants (souvent `AppButton`) doivent accepter `style` flex / width.
 */
export function ModalFooterActions({
  secondary,
  primary,
  hint,
  stacked,
}: ModalFooterActionsProps) {
  const { shouldStackRows } = useAccessibilityScale();
  const t = useResponsiveTokens();
  const stack = stacked ?? shouldStackRows;

  return (
    <View style={[styles.wrap, { gap: t.spacingSm }]}>
      {hint}
      <View
        style={[
          stack ? styles.column : styles.row,
          { gap: t.spacingSm },
        ]}
      >
        {secondary ? (
          <View style={stack ? styles.fullWidth : styles.flex}>{secondary}</View>
        ) : null}
        <View style={stack ? styles.fullWidth : styles.flex}>{primary}</View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: "100%",
  },
  row: {
    flexDirection: "row",
    alignItems: "stretch",
  },
  column: {
    flexDirection: "column",
    alignItems: "stretch",
  },
  flex: {
    flex: 1,
    minWidth: 0,
  },
  fullWidth: {
    width: "100%",
  },
});
