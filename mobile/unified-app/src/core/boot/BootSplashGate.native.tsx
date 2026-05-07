import type { ReactNode } from "react";
import LottieView from "lottie-react-native";
import { Animated, Platform } from "react-native";
import { useBootSplashGate } from "./useBootSplashGate";

type Props = { children: ReactNode };

/**
 * Overlay plein écran + Lottie (iOS / Android).
 */
export function BootSplashGate({ children }: Props) {
  const g = useBootSplashGate();

  return (
    <>
      {children}
      {g.overlayMounted ? (
        <Animated.View
          style={[
            g.styles.layer,
            {
              opacity: g.fadeOpacity,
            },
          ]}
          pointerEvents={g.pointerEvents}
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
                source={g.source}
                autoPlay
                loop={false}
                style={g.styles.lottie}
                resizeMode="contain"
                renderMode={Platform.OS === "android" ? "HARDWARE" : "AUTOMATIC"}
                onAnimationFinish={g.onLottieFinish}
              />
            </Animated.View>
          ) : null}
        </Animated.View>
      ) : null}
    </>
  );
}
