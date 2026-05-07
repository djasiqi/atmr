import type { ReactNode } from "react";
import { Animated, View } from "react-native";
import { useBootSplashGate } from "./useBootSplashGate";

type Props = { children: ReactNode };

/**
 * Overlay web : même timing / fondu que le natif, sans `lottie-react-native`
 * (évite la dépendance `@lottiefiles/dotlottie-react`). Zone centrale laissée vide ;
 * le fond blanc couvre l’écran jusqu’au fade-out.
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
              <View style={g.styles.lottie} />
            </Animated.View>
          ) : null}
        </Animated.View>
      ) : null}
    </>
  );
}
