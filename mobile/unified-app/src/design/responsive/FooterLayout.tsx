import { ReactNode, useMemo } from "react";
import { Platform, StyleSheet, View, ViewStyle } from "react-native";
import { useKeyboardHeight } from "./useKeyboardHeight";
import { useKeyboardLayout } from "../../features/chat/useKeyboardLayout";
import { useAppViewport } from "./useAppViewport";

export type FooterLayoutMode = "fixed" | "keyboardAware";

export type FooterLayoutProps = {
  /**
   * - `fixed` : pied de page collé en bas, respecte `safeBottom`. N'écoute pas le clavier.
   * - `keyboardAware` : pied de page collé en bas, se relève au-dessus du clavier (formulaires,
   *   panneaux d'action). **N'utilise pas** le pipeline chat (`bottomOffset` complet) — voir
   *   `ChatConversationShell` pour cela.
   */
  mode: FooterLayoutMode;
  children: ReactNode;
  /** Gap supplémentaire au-dessus de la safe area (par défaut tokens.spacingSm). */
  topGap?: number;
  /** Background du conteneur. */
  backgroundColor?: string;
  /** Bordure haute discrète (utile pour barres d'action). */
  withTopBorder?: boolean;
  /** Style libre — n'écrase pas le positionnement. */
  style?: ViewStyle;
  /** Force le pied à ne pas se relever (mode keyboardAware) — utile en test. */
  disableKeyboardOffset?: boolean;
};

/**
 * Pied de page sticky aligné sur le pipeline viewport/clavier de l'app.
 *
 * Important : le **mode `chat`** n'existe pas ici par design. La messagerie utilise
 * `ChatConversationShell` + `useKeyboardLayout` car la composeur (multi-ligne, audio,
 * pièces jointes, scroll feed) a des invariants spécifiques. Ne pas dupliquer.
 */
export function FooterLayout({
  mode,
  children,
  topGap,
  backgroundColor,
  withTopBorder = false,
  style,
  disableKeyboardOffset = false,
}: FooterLayoutProps) {
  const { bottomInset } = useAppViewport();
  const simple = useKeyboardHeight();
  const advanced = useKeyboardLayout();

  const computed = useMemo<ViewStyle>(() => {
    if (mode === "keyboardAware" && !disableKeyboardOffset) {
      // Sur iOS, KeyboardAvoidingView du Screen prend en charge le décalage du contenu ;
      // ici on n'aligne que la safe area + un padding bas quand le clavier est fermé.
      if (Platform.OS === "ios") {
        return {
          paddingBottom: simple.keyboardVisible ? 0 : bottomInset,
        };
      }
      // Android : suit `visibleBottomInset` qui combine resize + screenY + slack mesurés.
      return {
        paddingBottom: advanced.keyboardVisible
          ? advanced.visibleBottomInset
          : bottomInset,
      };
    }
    return { paddingBottom: bottomInset };
  }, [
    mode,
    disableKeyboardOffset,
    simple.keyboardVisible,
    advanced.keyboardVisible,
    advanced.visibleBottomInset,
    bottomInset,
  ]);

  return (
    <View
      pointerEvents="box-none"
      style={[
        styles.wrap,
        withTopBorder ? styles.border : null,
        backgroundColor ? { backgroundColor } : null,
        { paddingTop: topGap ?? 8 },
        computed,
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 16,
  },
  border: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(0,0,0,0.08)",
  },
});
