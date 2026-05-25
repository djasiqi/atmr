import { Animated } from "react-native";
import { Motion, MotionEasing } from "../navigationMotion";
import { resolveMotionDuration } from "../applyNavigationMotion";

/** Helper timing standard LIRIE — easing unique, native driver. */
export function runFadeSlideTiming(
  value: Animated.Value,
  toValue: number,
  durationMs: number = Motion.page,
  reduceMotion = false
): Animated.CompositeAnimation {
  return Animated.timing(value, {
    toValue,
    duration: resolveMotionDuration(durationMs, reduceMotion),
    easing: MotionEasing,
    useNativeDriver: true,
  });
}
