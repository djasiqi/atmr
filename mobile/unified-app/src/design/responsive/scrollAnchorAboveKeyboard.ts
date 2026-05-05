import type { MutableRefObject, RefObject } from "react";
import { Dimensions, Keyboard, Platform } from "react-native";
import type { KeyboardEvent } from "react-native";
import type { ScrollView, View } from "react-native";

const MARGIN_ABOVE_KB = 16;

/**
 * Haut du clavier en coordonnées écran.
 * Sur Android (Samsung, Huawei, etc.), `screenY` est souvent 0 ou incohérent ;
 * on déduit la position à partir de la hauteur fenêtre et de la hauteur clavier.
 */
function keyboardTopFromEvent(e: KeyboardEvent): number {
  const { height: windowH } = Dimensions.get("window");
  const { screenY, height: kbH } = e.endCoordinates;
  if (kbH <= 0) return windowH;

  const inferredTop = windowH - kbH;

  if (Platform.OS === "android") {
    if (screenY <= 1 || screenY >= windowH - 4) {
      return inferredTop;
    }
    return Math.min(screenY, inferredTop);
  }

  return screenY > 0 ? screenY : inferredTop;
}

function adjustOnce(
  scrollRef: RefObject<ScrollView | null>,
  scrollOffsetY: MutableRefObject<number>,
  anchorRef: RefObject<View | null>,
  e: KeyboardEvent,
): void {
  anchorRef.current?.measureInWindow((_x, y, _w, h) => {
    const keyboardTop = keyboardTopFromEvent(e);
    const bottom = y + h;
    if (bottom <= keyboardTop - MARGIN_ABOVE_KB) return;
    const overlap = bottom - (keyboardTop - MARGIN_ABOVE_KB);
    scrollRef.current?.scrollTo({
      y: scrollOffsetY.current + overlap,
      animated: true,
    });
  });
}

/**
 * Remonte la zone ciblée au-dessus du clavier si elle chevauche encore.
 * iOS : une passe après `keyboardWillShow`.
 * Android : plusieurs passes (resize OEM / Samsung Keyboard sont souvent retardés).
 */
export function scrollAnchorAboveKeyboard(
  scrollRef: RefObject<ScrollView | null>,
  scrollOffsetY: MutableRefObject<number>,
  anchorRef: RefObject<View | null>,
): void {
  if (Platform.OS === "web") return;

  const eventName = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";

  const sub = Keyboard.addListener(eventName, (e) => {
    sub.remove();

    const run = () => adjustOnce(scrollRef, scrollOffsetY, anchorRef, e);

    requestAnimationFrame(run);

    if (Platform.OS === "android") {
      setTimeout(run, 90);
      setTimeout(run, 220);
      setTimeout(run, 420);
      setTimeout(run, 600);
      setTimeout(run, 850);
    }
  });
}
