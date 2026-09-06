import { useEffect, useRef, useState } from "react";
import { Dimensions, Keyboard, Platform } from "react-native";

/** Plafond du sheet (écran plein), pas une hauteur forcée. */
export const CREATE_RIDE_SHEET_MAX_RATIO = 0.62;
export const CREATE_RIDE_SHEET_GUTTER = 8;
export const CREATE_RIDE_RESULTS_MIN_HEIGHT = 72;

export type CreateRideSheetLayoutInput = {
  windowHeight: number;
  keyboardHeight: number;
  resizedBySystem: boolean;
};

export type CreateRideSheetLayout = {
  availableHeight: number;
  /** Plafond uniquement — le sheet s’adapte au contenu en dessous. */
  maxSheetHeight: number;
  liftBottom: number;
  keyboardOpen: boolean;
};

export function computeCreateRideSheetLayout(
  input: CreateRideSheetLayoutInput
): CreateRideSheetLayout {
  const windowHeight = Math.max(0, input.windowHeight);
  const keyboardHeight = Math.max(0, input.keyboardHeight);
  const keyboardOpen = keyboardHeight > 0;
  const liftBottom = keyboardOpen && !input.resizedBySystem ? keyboardHeight : 0;
  const availableHeight = Math.max(240, windowHeight - liftBottom);
  const fullHeight = input.resizedBySystem ? windowHeight + keyboardHeight : windowHeight;
  const ratioCap = Math.round(Math.max(fullHeight, windowHeight) * CREATE_RIDE_SHEET_MAX_RATIO);
  return {
    availableHeight,
    maxSheetHeight: Math.max(
      240,
      Math.min(availableHeight - CREATE_RIDE_SHEET_GUTTER, ratioCap)
    ),
    liftBottom,
    keyboardOpen,
  };
}

/** Hauteur max de la liste : reste du plafond après chrome fixe. */
export function computeCreateRideResultsMaxHeight(
  maxSheetHeight: number,
  chromeHeight: number,
  footerHeight = 0
): number {
  return Math.max(
    CREATE_RIDE_RESULTS_MIN_HEIGHT,
    maxSheetHeight - Math.max(0, chromeHeight) - Math.max(0, footerHeight)
  );
}

/** Hauteur max d’une liste d’adresses au-dessus du clavier. */
export function computeCreateRideAddressListMaxHeight(
  input: CreateRideSheetLayoutInput,
  fallback = 230
): number {
  if (input.keyboardHeight <= 0) return fallback;
  const { availableHeight } = computeCreateRideSheetLayout(input);
  return Math.max(120, Math.min(fallback, Math.round(availableHeight * 0.36)));
}

const WINDOW_RESIZE_TRUST_THRESHOLD = 12;

/** Mesure locale (sans dépendre du chat) : hauteur fenêtre + clavier + resize système. */
export function useCreateRideKeyboardFrame(): CreateRideSheetLayoutInput & {
  keyboardVisible: boolean;
} {
  const [windowHeight, setWindowHeight] = useState(() => Dimensions.get("window").height);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const baselineRef = useRef(Dimensions.get("window").height);
  const keyboardOpenRef = useRef(false);

  useEffect(() => {
    const dimSub = Dimensions.addEventListener("change", ({ window }) => {
      setWindowHeight(window.height);
      if (!keyboardOpenRef.current) {
        baselineRef.current = window.height;
      }
    });
    if (Platform.OS === "web") {
      return () => dimSub?.remove();
    }
    const showEvent = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hideEvent = Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";
    const onShow = Keyboard.addListener(showEvent, (event) => {
      keyboardOpenRef.current = true;
      setKeyboardHeight(Math.max(0, event.endCoordinates.height));
      setWindowHeight(Dimensions.get("window").height);
    });
    const onHide = Keyboard.addListener(hideEvent, () => {
      keyboardOpenRef.current = false;
      const wh = Dimensions.get("window").height;
      baselineRef.current = wh;
      setKeyboardHeight(0);
      setWindowHeight(wh);
    });
    return () => {
      dimSub?.remove();
      onShow.remove();
      onHide.remove();
    };
  }, []);

  const resizedBySystem =
    keyboardHeight > 0 && baselineRef.current - windowHeight >= WINDOW_RESIZE_TRUST_THRESHOLD;

  return {
    windowHeight,
    keyboardHeight,
    resizedBySystem,
    keyboardVisible: keyboardHeight > 0,
  };
}
