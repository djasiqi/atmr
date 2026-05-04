import type { ReactNode, ReactElement } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  View,
  type StyleProp,
  type ViewStyle,
} from "react-native";
import { useResponsiveTokens } from "./useResponsiveTokens";
import { useAppViewport } from "./useAppViewport";

export type ScreenProps = {
  children: ReactNode;
  backgroundColor?: string;
  /** Si true, enveloppe le contenu dans un ScrollView (landing, formulaires longs). */
  scroll?: boolean;
  keyboardVerticalOffset?: number;
  contentContainerStyle?: StyleProp<ViewStyle>;
  /** Mettre false pour fond full-bleed ; combiner avec ResponsiveContainer pour le texte. */
  withHorizontalPadding?: boolean;
  safeTop?: boolean;
  safeBottom?: boolean;
  /**
   * Si false, le padding bas du ScrollView n’empile pas les insets safe area
   * (déjà pris en charge dans le contenu, ex. landing).
   */
  includeSafeAreaInScrollBottomPadding?: boolean;
  /** Ajouté au padding bas du contenu scroll (ex. barre flottante client). */
  extraScrollBottomPadding?: number;
  refreshControl?: ReactElement;
  showsVerticalScrollIndicator?: boolean;
  /**
   * iOS : avec `scroll={false}`, évite que le clavier masque les champs en bas d’écran.
   * Sans effet si `scroll={true}` (déjà couvert). Android ignoré — préférer resize natif.
   */
  keyboardAware?: boolean;
  /**
   * Android uniquement : enveloppe `KeyboardAvoidingView` — **opt-in** seulement si un écran
   * pose problème malgré `softwareKeyboardLayoutMode: "resize"` (risque double-offset si abus).
   */
  androidKeyboardFallback?: boolean;
};

/**
 * Enveloppe standard : flex 1, insets (safe area min 16 via useAppViewport).
 *
 * Clavier :
 * - **iOS** : `KeyboardAvoidingView` si `scroll` ou `keyboardAware`.
 * - **Android** : pas de KAV par défaut (resize fenêtre). `androidKeyboardFallback` pour cas exceptionnels.
 */
export function Screen({
  children,
  backgroundColor,
  scroll = false,
  keyboardVerticalOffset = 0,
  contentContainerStyle,
  withHorizontalPadding = true,
  safeTop = true,
  safeBottom = true,
  includeSafeAreaInScrollBottomPadding = true,
  extraScrollBottomPadding = 0,
  refreshControl,
  showsVerticalScrollIndicator = false,
  keyboardAware = false,
  androidKeyboardFallback = false,
}: ScreenProps) {
  const { topInset, bottomInset, horizontalPadding } = useAppViewport();
  const { scrollExtraBottomPadding, effectiveFontScale, fontScale } = useResponsiveTokens();

  const paddingTop = safeTop ? topInset : 0;
  const paddingBottom = safeBottom ? bottomInset : 0;
  const paddingHorizontal = withHorizontalPadding ? horizontalPadding : 0;

  const fontBoost = fontScale > 1 ? 8 * (fontScale - 1) : 0;
  const bottomScrollPad =
    (includeSafeAreaInScrollBottomPadding
      ? scrollExtraBottomPadding + (effectiveFontScale > 1 ? 6 * (effectiveFontScale - 1) : 0)
      : 12 + fontBoost) + extraScrollBottomPadding;

  const inner = scroll ? (
    <ScrollView
      keyboardShouldPersistTaps="handled"
      refreshControl={refreshControl}
      showsVerticalScrollIndicator={showsVerticalScrollIndicator}
      contentContainerStyle={[
        styles.scrollContent,
        {
          paddingBottom: bottomScrollPad,
          paddingHorizontal,
          flexGrow: 1,
        },
        contentContainerStyle,
      ]}
      style={styles.flex}
    >
      {children}
    </ScrollView>
  ) : (
    <View
      style={[
        styles.flex,
        {
          paddingHorizontal,
        },
      ]}
    >
      {children}
    </View>
  );

  const shell = (
    <View
      style={[
        styles.screen,
        {
          backgroundColor,
          paddingTop,
          paddingBottom,
        },
      ]}
    >
      {inner}
    </View>
  );

  const needsKavIos = Platform.OS === "ios" && (scroll || keyboardAware);
  const needsKavAndroid = Platform.OS === "android" && androidKeyboardFallback;
  const wrapKeyboardAvoiding = needsKavIos || needsKavAndroid;

  if (wrapKeyboardAvoiding) {
    return (
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.flex}
        keyboardVerticalOffset={keyboardVerticalOffset}
      >
        {shell}
      </KeyboardAvoidingView>
    );
  }

  return shell;
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  screen: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
});
