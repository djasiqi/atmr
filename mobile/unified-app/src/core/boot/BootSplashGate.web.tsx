import type { ReactNode } from "react";
import { useCallback, useEffect, useRef } from "react";
import { Animated, View } from "react-native";
import * as SplashScreen from "expo-splash-screen";
import { useBootSplashGate } from "./useBootSplashGate";
import { shouldReleaseNativeSplash } from "./hideNativeSplashWhenReady";
import { BootBrandMark } from "./BootBrandMark";

type Props = { children: ReactNode };

/**
 * Overlay web : même timing / fondu que le natif, sans `lottie-react-native`
 * (évite la dépendance `@lottiefiles/dotlottie-react`). Fond LIRIE + logo
 * jusqu’au fade-out — jamais #FFFFFF.
 */
export function BootSplashGate({ children }: Props) {
  const g = useBootSplashGate();
  const nativeHiddenRef = useRef(false);

  const releaseNativeSplash = useCallback((overlayLaidOut: boolean, overlayWillNeverShow: boolean) => {
    if (nativeHiddenRef.current) return;
    if (!shouldReleaseNativeSplash({ overlayLaidOut, overlayWillNeverShow })) return;
    nativeHiddenRef.current = true;
    void SplashScreen.hideAsync().catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!g.overlayMounted && !g.showOverlay) {
      releaseNativeSplash(false, true);
    }
  }, [g.overlayMounted, g.showOverlay, releaseNativeSplash]);

  return (
    <>
      {children}
      {g.overlayMounted ? (
        <Animated.View
          onLayout={() => releaseNativeSplash(true, false)}
          style={[
            g.styles.layer,
            {
              opacity: g.fadeOpacity,
              pointerEvents: g.pointerEvents,
            },
          ]}
        >
          <BootBrandMark />
          {g.showLottieLayer ? (
            <Animated.View
              style={[
                g.styles.lottieLayer,
                {
                  opacity: g.lottieOpacity,
                },
              ]}
            >
              <View style={g.lottieStyle} />
            </Animated.View>
          ) : null}
        </Animated.View>
      ) : null}
    </>
  );
}
