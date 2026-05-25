import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Dimensions, Keyboard, PixelRatio, Platform } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import {
  computeVisibleBottomInsets,
  shellFooterOffset,
  type VisibleBottomMetrics,
} from "./keyboardLayoutMetrics";

const LOG_KEYBOARD_LAYOUT =
  __DEV__ &&
  Platform.OS === "android" &&
  process.env.EXPO_PUBLIC_CHAT_LAYOUT_DEBUG === "1";

type KeyboardSession = {
  keyboardHeight: number;
  screenY: number;
  windowHeightAtShow: number;
};

export type KeyboardLayout = {
  keyboardVisible: boolean;
  keyboardHeight: number;
  visibleHeight: number;
  baselineWindowHeight: number;
  screenY: number | null;
  resizeDelta: number;
  /** Vrai quand le système a effectivement redimensionné la fenêtre (Android adjustResize fiable). */
  isResizedBySystem: boolean;
  visibleBottomInset: number;
  safeBottom: number;
  footerOffset: (shellBottomGap?: number) => number;
  /** Alias sémantique de {@link footerOffset} — usage hors chat. */
  bottomOffset: (shellBottomGap?: number) => number;
  platform: typeof Platform.OS;
  metrics: VisibleBottomMetrics | null;
};

/** Seuil de confiance sur le resize fenêtre (cf. keyboardLayoutMetrics). */
const WINDOW_RESIZE_TRUST_THRESHOLD = 12;

const KeyboardLayoutContext = createContext<KeyboardLayout | null>(null);

function useKeyboardLayoutState(disabled = false): KeyboardLayout {
  const insets = useSafeAreaInsets();
  const [session, setSession] = useState<KeyboardSession | null>(null);
  const [windowHeight, setWindowHeight] = useState(() => Dimensions.get("window").height);
  const baselineRef = useRef(Dimensions.get("window").height);
  const sessionOpenRef = useRef(false);

  useEffect(() => {
    if (disabled) return undefined;
    const dimSub = Dimensions.addEventListener("change", ({ window }) => {
      setWindowHeight(window.height);
      if (!sessionOpenRef.current) {
        baselineRef.current = window.height;
      }
    });
    return () => dimSub?.remove();
  }, [disabled]);

  useEffect(() => {
    if (disabled) return undefined;
    if (Platform.OS === "web") return undefined;

    const showEvent = Platform.OS === "ios" ? "keyboardWillShow" : "keyboardDidShow";
    const hideEvent = Platform.OS === "ios" ? "keyboardWillHide" : "keyboardDidHide";

    const onShow = Keyboard.addListener(showEvent, (event) => {
      const keyboardHeight = Math.max(0, event.endCoordinates.height);
      const wh = Dimensions.get("window").height;
      const screenY = event.endCoordinates.screenY;

      setSession({
        keyboardHeight,
        screenY,
        windowHeightAtShow: baselineRef.current,
      });
      sessionOpenRef.current = true;
      setWindowHeight(wh);

      if (LOG_KEYBOARD_LAYOUT) {
        const preview = computeVisibleBottomInsets({
          baselineWindowHeight: baselineRef.current,
          windowHeight: wh,
          keyboardHeight,
          screenY,
        });
        console.log("[chat-keyboard] didShow", {
          baselineWindow: baselineRef.current,
          windowHeight: wh,
          keyboardHeight,
          screenY,
          fontScale: PixelRatio.getFontScale(),
          pixelRatio: PixelRatio.get(),
          ...preview,
        });
      }

      requestAnimationFrame(() => {
        const whAfter = Dimensions.get("window").height;
        setWindowHeight(whAfter);
        if (LOG_KEYBOARD_LAYOUT && whAfter !== wh) {
          console.log("[chat-keyboard] window resize after show", {
            windowHeight: whAfter,
            resizeDelta: Math.max(0, baselineRef.current - whAfter),
          });
        }
      });
    });

    const onHide = Keyboard.addListener(hideEvent, () => {
      const wh = Dimensions.get("window").height;
      sessionOpenRef.current = false;
      baselineRef.current = wh;
      setSession(null);
      setWindowHeight(wh);
      if (LOG_KEYBOARD_LAYOUT) {
        console.log("[chat-keyboard] didHide", { windowHeight: wh });
      }
    });

    return () => {
      onShow.remove();
      onHide.remove();
    };
  }, [disabled]);

  const keyboardVisible = session != null;

  const metrics = useMemo(() => {
    if (session == null) return null;
    return computeVisibleBottomInsets({
      baselineWindowHeight: session.windowHeightAtShow,
      windowHeight,
      keyboardHeight: session.keyboardHeight,
      screenY: session.screenY,
    });
  }, [session, windowHeight]);

  const visibleBottomInset =
    keyboardVisible && Platform.OS === "android" && metrics != null
      ? metrics.visibleBottomInset
      : 0;

  const footerOffset = useMemo(
    () => (shellBottomGap = 0) => shellFooterOffset(visibleBottomInset, shellBottomGap),
    [visibleBottomInset]
  );

  const resizeDelta = metrics?.resizeDelta ?? 0;
  const isResizedBySystem = resizeDelta >= WINDOW_RESIZE_TRUST_THRESHOLD;

  return {
    keyboardVisible,
    keyboardHeight: session?.keyboardHeight ?? 0,
    visibleHeight: windowHeight,
    baselineWindowHeight: baselineRef.current,
    screenY: session?.screenY ?? null,
    resizeDelta,
    isResizedBySystem,
    visibleBottomInset,
    safeBottom: keyboardVisible ? 0 : insets.bottom,
    footerOffset,
    bottomOffset: footerOffset,
    platform: Platform.OS,
    metrics,
  };
}

/** Fournit une instance unique du layout clavier pour feed + composeur. */
export function KeyboardLayoutProvider({ children }: { children: ReactNode }) {
  const layout = useKeyboardLayoutState();
  return (
    <KeyboardLayoutContext.Provider value={layout}>{children}</KeyboardLayoutContext.Provider>
  );
}

/**
 * Layout clavier LIRIE — mesure le bas visible, pas une hauteur OEM fixe.
 * Doit être sous {@link KeyboardLayoutProvider} dans les écrans chat.
 */
export function useKeyboardLayout(): KeyboardLayout {
  const ctx = useContext(KeyboardLayoutContext);
  const fallback = useKeyboardLayoutState(ctx != null);
  return ctx ?? fallback;
}

