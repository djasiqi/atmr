import React, { useEffect, useMemo, useRef, type ReactNode, type ReactElement, type RefObject } from "react";
import {
  Animated,
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
import { usePathname } from "expo-router";
import { useResponsiveTokens } from "./useResponsiveTokens";
import { useAppViewport } from "./useAppViewport";
import { Motion, MotionEasing } from "../navigation/navigationMotion";
import { resolveMotionDuration } from "../navigation/applyNavigationMotion";
import { useReduceMotion } from "../navigation/useReduceMotion";

export type ScreenProps = {
  children: ReactNode;
  backgroundColor?: string;
  /** Si true, enveloppe le contenu dans un ScrollView (landing, formulaires longs). */
  scroll?: boolean;
  /**
   * Bandeau fixe au-dessus du contenu (ex. en-tête entreprise).
   * Avec `scroll={true}` : reste visible pendant le défilement.
   * Avec `scroll={false}` : en-tête fixe au-dessus du corps (ex. FlatList interne).
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
  /**
   * Animation de transition de page interne (fade + slide 8 px, 180 ms).
   * Désactivée par défaut depuis le contrat LIRIE :
   *   - écrans sous tabs → animation portée par `floatingTabScreenOptions`
   *   - écrans dans un stack → animation portée par `stackScreenOptions`
   * À n'activer (true) que pour un écran isolé hors tabs/stack.
   */
  pageTransition?: boolean;
};

const TAB_ORDER = [
  "dashboard",
  "rides",
  "chat",
  "clients-facturation",
  "settings",
  "index",
  "trips",
  "missions",
  "schedule",
  "profile",
] as const;
let lastRouteLeaf: string | null = null;

function extractRouteLeaf(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  return parts.at(-1) ?? "";
}

function computeSlideDirection(previousLeaf: string | null, nextLeaf: string): 1 | -1 {
  if (!previousLeaf) return 1;
  const prevIdx = TAB_ORDER.indexOf(previousLeaf as (typeof TAB_ORDER)[number]);
  const nextIdx = TAB_ORDER.indexOf(nextLeaf as (typeof TAB_ORDER)[number]);
  if (prevIdx < 0 || nextIdx < 0) return 1;
  return nextIdx >= prevIdx ? 1 : -1;
}

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
  pageTransition = false,
}: ScreenProps) {
  const { topInset, bottomInset, horizontalPadding } = useAppViewport();
  const { scrollExtraBottomPadding, effectiveFontScale, fontScale } = useResponsiveTokens();
  const pathname = usePathname();
  const routeLeaf = useMemo(() => extractRouteLeaf(pathname), [pathname]);
  const transitionProgress = useRef(new Animated.Value(pageTransition ? 0 : 1)).current;
  const reduceMotion = useReduceMotion();

  useEffect(() => {
    if (!pageTransition) {
      transitionProgress.setValue(1);
      lastRouteLeaf = routeLeaf;
      return;
    }
    transitionProgress.setValue(0);
    Animated.timing(transitionProgress, {
      toValue: 1,
      duration: resolveMotionDuration(Motion.page, reduceMotion),
      easing: MotionEasing,
      useNativeDriver: true,
    }).start();
    lastRouteLeaf = routeLeaf;
  }, [pageTransition, routeLeaf, transitionProgress, reduceMotion]);

  const direction = computeSlideDirection(lastRouteLeaf, routeLeaf);
  const transitionStyle = pageTransition
    ? {
        opacity: transitionProgress.interpolate({
          inputRange: [0, 1],
          outputRange: [0.96, 1],
        }),
        transform: [
          {
            translateX: transitionProgress.interpolate({
              inputRange: [0, 1],
              outputRange: [direction > 0 ? 8 : -8, 0],
            }),
          },
        ],
      }
    : undefined;

  /** Avec sticky header, la safe area haute est gérée dans le bandeau (fond pleine largeur sous la barre système). */
  const shellPaddingTop = stickyHeader != null && safeTop ? 0 : safeTop ? topInset : 0;
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

  const staticBody = (
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
  ) : stickyHeader != null ? (
    <View style={styles.flex}>
      <View style={styles.stickyHeaderSlot}>{stickyRendered}</View>
      {staticBody}
    </View>
  ) : (
    staticBody
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
      {pageTransition ? (
        <Animated.View style={[styles.flex, transitionStyle]}>{inner}</Animated.View>
      ) : (
        inner
      )}
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
