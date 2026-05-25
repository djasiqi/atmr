import { ReactNode, useMemo } from "react";
import { StyleSheet, View, type ViewStyle } from "react-native";
import { useAppViewport } from "./useAppViewport";
import { useResponsiveTokens } from "./useResponsiveTokens";

export type BottomSheetLayoutMetrics = {
  cardMaxHeight: number;
  scrollMaxHeight: number;
  paddingBottom: number;
};

export type BottomSheetLayoutOptions = {
  /** Header, CTA, margins, or any fixed chrome outside the scrollable body. */
  reservedChromeHeight?: number;
  maxHeightRatio?: number;
  maxHeightCap?: number;
  minScrollHeight?: number;
  bottomPaddingExtra?: number;
};

export type BottomSheetLayoutProps = BottomSheetLayoutOptions & {
  children: ReactNode;
  style?: ViewStyle;
};

export function computeBottomSheetLayout(
  usableHeight: number,
  bottomInset: number,
  tokens: Pick<ReturnType<typeof useResponsiveTokens>, "modalSheetMaxHeightRatio" | "modalSheetMaxHeightCap">,
  options: BottomSheetLayoutOptions = {}
): BottomSheetLayoutMetrics {
  const ratio = options.maxHeightRatio ?? tokens.modalSheetMaxHeightRatio;
  const cap = options.maxHeightCap ?? tokens.modalSheetMaxHeightCap;
  const reservedChromeHeight = options.reservedChromeHeight ?? 0;
  const minScrollHeight = options.minScrollHeight ?? 120;
  const paddingBottom = Math.max(16, bottomInset + (options.bottomPaddingExtra ?? 8));
  const cardMaxHeight = Math.min(Math.round(usableHeight * ratio), cap);
  const scrollMaxHeight = Math.max(
    minScrollHeight,
    cardMaxHeight - reservedChromeHeight - paddingBottom
  );

  return { cardMaxHeight, scrollMaxHeight, paddingBottom };
}

export function useBottomSheetLayout(options: BottomSheetLayoutOptions = {}): BottomSheetLayoutMetrics {
  const { usableHeight, bottomInset } = useAppViewport();
  const tokens = useResponsiveTokens();

  return useMemo(
    () => computeBottomSheetLayout(usableHeight, bottomInset, tokens, options),
    [usableHeight, bottomInset, tokens, options]
  );
}

/** Generic absolute-bottom sheet container for simple modal bodies. */
export function BottomSheetLayout({ children, style, ...options }: BottomSheetLayoutProps) {
  const metrics = useBottomSheetLayout(options);

  return (
    <View
      style={[
        styles.card,
        { maxHeight: metrics.cardMaxHeight, paddingBottom: metrics.paddingBottom },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    width: "100%",
  },
});
