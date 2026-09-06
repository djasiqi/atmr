import type { ReactNode } from "react";
import { useCallback, useEffect, useRef } from "react";
import LottieView from "lottie-react-native";
import { Animated, Platform } from "react-native";
import * as SplashScreen from "expo-splash-screen";
import { useBootSplashGate } from "./useBootSplashGate";
import { shouldReleaseNativeSplash } from "./hideNativeSplashWhenReady";
import { BootBrandMark } from "./BootBrandMark";

type Props = { children: ReactNode };

/**
 * Overlay plein écran : fond LIRIE + logo (wordmark toujours,
 * Lottie par-dessus au 1er launch). iOS / Android.
 */
export function BootSplashGate({ children }: Props) {
  const g = useBootSplashGate();
  const lottieRef = useRef<LottieView>(null);
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

  // Android + New Architecture : `autoPlay` ne démarre pas toujours l'animation.
  // On force la lecture via la ref (non bloquant : la sortie du splash est de toute
  // façon garantie par le timeout dans useBootSplashGate).
  useEffect(() => {
    if (Platform.OS !== "android" || !g.showLottieLayer) {
      return;
    }
    const id = setTimeout(() => {
      lottieRef.current?.play();
    }, 150);
    return () => clearTimeout(id);
  }, [g.showLottieLayer]);

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
              <LottieView
                ref={lottieRef}
                source={g.source}
                autoPlay
                loop={false}
                style={g.lottieStyle}
                resizeMode="cover"
                renderMode={Platform.OS === "android" ? "SOFTWARE" : "AUTOMATIC"}
                onAnimationFinish={g.onLottieFinish}
              />
            </Animated.View>
          ) : null}
        </Animated.View>
      ) : null}
    </>
  );
}
