import { useEffect, useRef, type ReactNode } from "react";
import { Animated, StyleSheet } from "react-native";
import { resolveMotionDuration } from "../applyNavigationMotion";
import { startMotionAnimation, type MotionLayer } from "../motionKpi";
import { Motion, MotionDistance, MotionEasing } from "../navigationMotion";
import { useReduceMotion } from "../useReduceMotion";

export type DataRevealProps = {
  children: ReactNode;
  /** Passe true quand le contenu réel remplace skeleton / vide. */
  visible: boolean;
  layer?: MotionLayer;
  kind?: string;
  screen?: string;
  style?: object;
};

/**
 * Fondu discret skeleton → contenu (Motion.data, 150 ms).
 * Une seule couche opacity — pas de slide.
 */
export function DataReveal({
  children,
  visible,
  layer = "data",
  kind = "data_reveal",
  screen,
  style,
}: DataRevealProps) {
  const reduceMotion = useReduceMotion();
  const opacity = useRef(new Animated.Value(visible ? MotionDistance.dataFadeTo : MotionDistance.dataFadeFrom)).current;
  const endKpiRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!visible) {
      opacity.setValue(MotionDistance.dataFadeFrom);
      return;
    }
    endKpiRef.current?.();
    endKpiRef.current = startMotionAnimation({
      layer,
      kind,
      duration_expected_ms: Motion.data,
      screen,
    });
    const duration = resolveMotionDuration(Motion.data, reduceMotion);
    Animated.timing(opacity, {
      toValue: MotionDistance.dataFadeTo,
      duration,
      easing: MotionEasing,
      useNativeDriver: true,
    }).start(({ finished }) => {
      if (finished) {
        endKpiRef.current?.();
        endKpiRef.current = null;
      }
    });
    return () => {
      endKpiRef.current?.();
      endKpiRef.current = null;
    };
  }, [visible, reduceMotion, opacity, layer, kind, screen]);

  return (
    <Animated.View style={[styles.flex, style, { opacity }]} pointerEvents={visible ? "auto" : "none"}>
      {children}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
});
