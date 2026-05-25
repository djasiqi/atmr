import { Animated, type ViewStyle } from "react-native";
import { MotionDistance } from "./navigationMotion";

type TabSceneInterpolationProps = {
  current: {
    progress: Animated.Value;
  };
};

/**
 * Transition LIRIE pour onglets flottants : fade discret + slide 8 px.
 * Pas de scale (évite effet zoom). `reduceMotion=true` → uniquement opacity.
 */
export function lirieTabFadeSlide(
  pageBg: string,
  reduceMotion = false
): (props: TabSceneInterpolationProps) => { sceneStyle: Animated.WithAnimatedValue<ViewStyle> } {
  const slide = reduceMotion ? 0 : MotionDistance.tabSlidePx;
  return ({ current }) => {
    const opacity = current.progress.interpolate({
      inputRange: [-1, 0, 1],
      outputRange: [MotionDistance.fadeFrom, MotionDistance.fadeTo, MotionDistance.fadeFrom],
      extrapolate: "clamp",
    });
    const translateX = current.progress.interpolate({
      inputRange: [-1, 0, 1],
      outputRange: [-slide, 0, slide],
      extrapolate: "clamp",
    });

    return {
      sceneStyle: {
        flex: 1,
        backgroundColor: pageBg,
        overflow: "hidden",
        opacity,
        transform: [{ translateX }],
      },
    };
  };
}
