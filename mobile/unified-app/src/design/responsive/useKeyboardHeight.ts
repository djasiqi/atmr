import { useEffect, useState } from "react";
import { Keyboard, Platform } from "react-native";
import { useResponsiveTokens } from "./useResponsiveTokens";

export type KeyboardHeightState = {
  keyboardVisible: boolean;
  keyboardHeight: number;
  scrollPaddingBottom: number;
};

/**
 * Clavier léger pour formulaires / scroll public.
 * Ne pas utiliser dans le chat — préférer {@link useKeyboardLayout}.
 */
export function useKeyboardHeight(): KeyboardHeightState {
  const { keyboardScrollPaddingMin, keyboardScrollPaddingExtra } = useResponsiveTokens();
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const [scrollPaddingBottom, setScrollPaddingBottom] = useState(0);

  useEffect(() => {
    if (Platform.OS === "web") return undefined;

    const showEvent = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hideEvent = Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";

    const onShow = Keyboard.addListener(showEvent, (e) => {
      const h = Math.max(0, e.endCoordinates?.height ?? 0);
      const computed =
        h > 0 ? Math.round(h + keyboardScrollPaddingExtra) : keyboardScrollPaddingMin;
      setKeyboardHeight(h);
      setScrollPaddingBottom(Math.max(keyboardScrollPaddingMin, computed));
      setKeyboardVisible(true);
    });

    const onHide = Keyboard.addListener(hideEvent, () => {
      setKeyboardVisible(false);
      setKeyboardHeight(0);
      setScrollPaddingBottom(0);
    });

    return () => {
      onShow.remove();
      onHide.remove();
    };
  }, [keyboardScrollPaddingMin, keyboardScrollPaddingExtra]);

  return { keyboardVisible, keyboardHeight, scrollPaddingBottom };
}
