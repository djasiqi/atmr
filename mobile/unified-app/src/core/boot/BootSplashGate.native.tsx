import type { ReactNode } from "react";
import { useEffect, useRef } from "react";
import LottieView from "lottie-react-native";
import { Animated, Platform } from "react-native";
import { useBootSplashGate } from "./useBootSplashGate";

type Props = { children: ReactNode };

/**
 * Overlay plein écran + Lottie (iOS / Android).
 */
export function BootSplashGate({ children }: Props) {
  const g = useBootSplashGate();
  const lottieRef = useRef<LottieView>(null);

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
          style={[
            g.styles.layer,
            {
              opacity: g.fadeOpacity,
              pointerEvents: g.pointerEvents,
            },
          ]}
        >
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
