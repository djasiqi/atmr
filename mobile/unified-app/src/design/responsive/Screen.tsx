import React, { type ReactNode, type ReactElement, type RefObject } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  View,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
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
  /**
   * Bandeau fixe au-dessus du scroll (ex. en-tête entreprise).
   * uniquement avec `scroll={true}` : reste visible pendant le défilement.
   */
  stickyHeader?: ReactNode;
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
  /**
   * iOS (ScrollView) : ajuste `contentInset` / indicateurs quand le clavier est ouvert.
   * Activable par écran pour les formulaires longs ou landing scrollable.
   */
  automaticallyAdjustKeyboardInsets?: boolean;
  /** Référence au `ScrollView` interne lorsque `scroll` est activé (scroll programmatique, tests). */
  scrollViewRef?: RefObject<ScrollView | null>;
  /** Uniquement si `scroll` : propagé au `ScrollView`. */
  onScroll?: (event: NativeSyntheticEvent<NativeScrollEvent>) => void;
  scrollEventThrottle?: number;
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
  automaticallyAdjustKeyboardInsets = false,
  scrollViewRef,
  onScroll,
  scrollEventThrottle,
  stickyHeader,
}: ScreenProps) {
  const { topInset, bottomInset, horizontalPadding } = useAppViewport();
  const { scrollExtraBottomPadding, effectiveFontScale, fontScale } = useResponsiveTokens();

  /** Avec sticky header, la safe area haute est gérée dans le bandeau (fond pleine largeur sous la barre système). */
  const shellPaddingTop =
    stickyHeader != null && scroll && safeTop ? 0 : safeTop ? topInset : 0;
  const paddingBottom = safeBottom ? bottomInset : 0;
  const paddingHorizontal = withHorizontalPadding ? horizontalPadding : 0;

  const fontBoost = fontScale > 1 ? 8 * (fontScale - 1) : 0;
  const bottomScrollPad =
    (includeSafeAreaInScrollBottomPadding
      ? scrollExtraBottomPadding + (effectiveFontScale > 1 ? 6 * (effectiveFontScale - 1) : 0)
      : 12 + fontBoost) + extraScrollBottomPadding;

  const scrollBody = (
    <ScrollView
      ref={scrollViewRef}
      keyboardShouldPersistTaps="handled"
      automaticallyAdjustKeyboardInsets={automaticallyAdjustKeyboardInsets}
      onScroll={onScroll}
      scrollEventThrottle={scrollEventThrottle}
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
  );

  const stickyRendered =
    stickyHeader != null && React.isValidElement(stickyHeader)
      ? React.cloneElement(stickyHeader as React.ReactElement<{ topSafeAreaPx?: number }>, {
          topSafeAreaPx: safeTop ? topInset : 0,
        })
      : stickyHeader;

  const inner = scroll ? (
    stickyHeader != null ? (
      <View style={styles.flex}>
        {/*
          Bandeau full-bleed : pas de padding ici (sinon fond blanc / bordure ne vont pas bord à bord).
          Le composant sticky (ex. EnterpriseHeader) applique le même gutter que le scroll en interne.
        */}
        <View style={styles.stickyHeaderSlot}>{stickyRendered}</View>
        {scrollBody}
      </View>
    ) : (
      scrollBody
    )
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
          paddingTop: shellPaddingTop,
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
  /** Largeur 100 % pour que l’en-tête colle aux bords (web + safe areas gérées dans le composant). */
  stickyHeaderSlot: {
    alignSelf: "stretch",
    width: "100%",
  },
  screen: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
});
