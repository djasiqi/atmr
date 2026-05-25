import { Platform, type ViewStyle } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useResponsiveTokens } from "../../design/responsive/useResponsiveTokens";
import { COMPOSER_EDGE_GAP, computeOemSafetyMargin } from "./keyboardLayoutMetrics";
import { useKeyboardLayout } from "./useKeyboardLayout";

const LOG_KEYBOARD_LAYOUT =
  __DEV__ &&
  Platform.OS === "android" &&
  process.env.EXPO_PUBLIC_CHAT_LAYOUT_DEBUG === "1";

/** @deprecated Préférer {@link KeyboardLayout} depuis {@link useKeyboardLayout}. */
export type KeyboardFrame = {
  keyboardHeight: number;
  screenY: number;
  resizeDelta: number;
  liftFromBottom: number;
};

/** @deprecated Préférer {@link useKeyboardLayout}. */
export function useKeyboardFrame(): KeyboardFrame | null {
  const layout = useKeyboardLayout();
  if (!layout.keyboardVisible || layout.screenY == null) return null;
  return {
    keyboardHeight: layout.keyboardHeight,
    screenY: layout.screenY,
    resizeDelta: layout.resizeDelta,
    liftFromBottom: layout.visibleBottomInset,
  };
}

/** @deprecated Préférer {@link useKeyboardLayout}. */
export function useKeyboardBottomInset(): number {
  return useKeyboardLayout().visibleBottomInset;
}

type ChatFooterStyleOptions = {
  horizontalPadding: number;
  backgroundColor?: string;
  shellBottomGap?: number;
};

export type ChatFooterLayout = {
  positionStyle: ViewStyle;
  bottomOffset: number;
};

/**
 * Pied de fil en position absolue (bas de la scène).
 *
 * **Repos** : `bottom: 0`, gap 8 px.
 * **Clavier Android** : offset depuis {@link useKeyboardLayout} (mesure espace visible).
 * **iOS** : `Screen keyboardAware` — pas de lift manuel.
 */
export function useChatFooterLayout({
  horizontalPadding,
  backgroundColor = "transparent",
  shellBottomGap = 0,
}: ChatFooterStyleOptions): ChatFooterLayout {
  const { footerOffset, keyboardVisible, visibleBottomInset } = useKeyboardLayout();
  const { spacingMd, fontScale } = useResponsiveTokens();
  const insets = useSafeAreaInsets();
  const gap = Platform.OS === "web" ? spacingMd : COMPOSER_EDGE_GAP;

  const bottomOffset =
    Platform.OS === "android" && keyboardVisible
      ? footerOffset(shellBottomGap) + computeOemSafetyMargin(fontScale)
      : Platform.OS === "ios" && !keyboardVisible
        ? insets.bottom
        : 0;

  if (LOG_KEYBOARD_LAYOUT && keyboardVisible) {
    console.log("[chat-keyboard] footer offset", {
      visibleBottomInset,
      shellBottomGap,
      bottomOffset,
    });
  }

  return {
    bottomOffset,
    positionStyle: {
      position: "absolute",
      left: 0,
      right: 0,
      bottom: bottomOffset,
      paddingHorizontal: horizontalPadding,
      paddingTop: 2,
      paddingBottom: gap,
      backgroundColor,
      gap: 2,
    },
  };
}

/** @deprecated Préférer {@link useChatFooterLayout}. */
export function useChatFooterPositionStyle(options: ChatFooterStyleOptions): ViewStyle {
  return useChatFooterLayout(options).positionStyle;
}

/** @deprecated Préférer {@link useChatFooterLayout} + {@link ChatConversationShell}. */
export function useChatFooterStyle(options: ChatFooterStyleOptions): ViewStyle {
  return useChatFooterLayout(options).positionStyle;
}

/** @deprecated */
export function useChatComposerBottomPadding(restingGap: number): number {
  const { visibleBottomInset, keyboardVisible } = useKeyboardLayout();
  if (keyboardVisible && Platform.OS === "android") {
    return restingGap + visibleBottomInset;
  }
  return restingGap;
}

