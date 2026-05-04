import { useMemo } from "react";
import { useWindowDimensions } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

/** Padding horizontal global par bucket (tiny / compact / regular / tablet). */
const GUTTER_TINY = 14;
const GUTTER_COMPACT = 18;
const GUTTER_REGULAR = 22;
const GUTTER_TABLET = 32;

/** Plafond lisibilité pour la colonne de contenu (pas une largeur minimale qui overflow). */
const CONTENT_CAP_PHONE = 440;
const CONTENT_CAP_TABLET = 520;

export type AppViewport = {
  width: number;
  height: number;
  shortest: number;
  longest: number;
  isLandscape: boolean;
  isTiny: boolean;
  isCompact: boolean;
  isRegular: boolean;
  isTablet: boolean;
  safeTop: number;
  safeBottom: number;
  safeLeft: number;
  safeRight: number;
  /** Alias identiques à safeTop / safeBottom (API unique). */
  topInset: number;
  bottomInset: number;
  usableHeight: number;
  usableWidth: number;
  horizontalPadding: number;
  contentWidth: number;
};

type RawInsets = { top: number; bottom: number; left: number; right: number };

/** Logique pure : tests unitaires + une seule vérité géométrique. */
export function computeAppViewport(width: number, height: number, rawInsets: RawInsets): AppViewport {
  const shortest = Math.min(width, height);
  const longest = Math.max(width, height);
  const isLandscape = width > height;

  const isTiny = shortest < 360;
  const isCompact = shortest >= 360 && shortest < 400;
  const isRegular = shortest >= 400 && shortest < 768;
  const isTablet = shortest >= 768;

  const safeTop = Math.max(rawInsets.top, 16);
  const safeBottom = Math.max(rawInsets.bottom, 16);
  const safeLeft = Math.max(rawInsets.left, 0);
  const safeRight = Math.max(rawInsets.right, 0);

  const usableWidth = Math.max(0, width - safeLeft - safeRight);
  const usableHeight = Math.max(0, height - safeTop - safeBottom);

  const horizontalPadding = isTablet
    ? GUTTER_TABLET
    : isTiny
      ? GUTTER_TINY
      : isCompact
        ? GUTTER_COMPACT
        : GUTTER_REGULAR;

  const rawContentWidth = usableWidth - 2 * horizontalPadding;
  const cap = isTablet ? CONTENT_CAP_TABLET : CONTENT_CAP_PHONE;
  const contentWidth = Math.min(Math.max(rawContentWidth, 0), cap);

  return {
    width,
    height,
    shortest,
    longest,
    isLandscape,
    isTiny,
    isCompact,
    isRegular,
    isTablet,
    safeTop,
    safeBottom,
    safeLeft,
    safeRight,
    topInset: safeTop,
    bottomInset: safeBottom,
    usableHeight,
    usableWidth,
    horizontalPadding,
    contentWidth,
  };
}

export function useAppViewport(): AppViewport {
  const { width, height } = useWindowDimensions();
  const rawInsets = useSafeAreaInsets();

  return useMemo(
    () => computeAppViewport(width, height, rawInsets),
    [width, height, rawInsets]
  );
}
