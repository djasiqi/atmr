import { useEffect, useRef } from "react";
import { Animated, StyleSheet, View } from "react-native";
import { Motion, MotionEasing } from "./navigationMotion";
import { resolveMotionDuration } from "./applyNavigationMotion";
import { startMotionAnimation } from "./motionKpi";
import { useReduceMotion } from "./useReduceMotion";

type TabRect = { x: number; width: number };

export type AnimatedTabIndicatorProps = {
  /** Index actif. */
  activeIndex: number;
  /** Mesures `onLayout` pour chaque onglet. */
  rects: readonly (TabRect | null)[];
  /** Hauteur du trait (par défaut 3). */
  height?: number;
  /** Couleur du trait (par défaut teal LIRIE). */
  color?: string;
  /** Identifiant facultatif pour la télémétrie. */
  screen?: string;
};

/**
 * Indicateur teal glissant sous l'onglet actif — `Animated.timing` LIRIE, 180 ms, pas de spring.
 * Reste invisible tant que les rectangles ne sont pas mesurés (`onLayout` initial).
 */
export function AnimatedTabIndicator({
  activeIndex,
  rects,
  height = 3,
  color = "#0A8F7A",
  screen,
}: AnimatedTabIndicatorProps) {
  const translateX = useRef(new Animated.Value(0)).current;
  const width = useRef(new Animated.Value(0)).current;
  const reduceMotion = useReduceMotion();
  const measuredRef = useRef(false);

  useEffect(() => {
    const target = rects[activeIndex];
    if (!target) return;
    const duration = resolveMotionDuration(Motion.page, reduceMotion);
    if (!measuredRef.current) {
      translateX.setValue(target.x);
      width.setValue(target.width);
      measuredRef.current = true;
      return;
    }
    const end = startMotionAnimation({
      layer: "inbox_indicator",
      kind: "indicator_slide",
      duration_expected_ms: Motion.page,
      screen,
      source: "motion.inbox_indicator",
    });
    Animated.parallel([
      Animated.timing(translateX, {
        toValue: target.x,
        duration,
        easing: MotionEasing,
        useNativeDriver: false,
      }),
      Animated.timing(width, {
        toValue: target.width,
        duration,
        easing: MotionEasing,
        useNativeDriver: false,
      }),
    ]).start(({ finished }) => {
      if (finished) end();
    });
  }, [activeIndex, rects, translateX, width, reduceMotion, screen]);

  return (
    <View pointerEvents="none" style={styles.layer}>
      <Animated.View
        style={[
          styles.bar,
          {
            height,
            backgroundColor: color,
            transform: [{ translateX }],
            width,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  layer: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    height: 3,
  },
  bar: {
    position: "absolute",
    left: 0,
    bottom: 0,
    borderTopLeftRadius: 2,
    borderTopRightRadius: 2,
  },
});
