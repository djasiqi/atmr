import { useEffect } from "react";
import { StyleSheet, View } from "react-native";
import Animated, {
  Easing,
  type SharedValue,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withRepeat,
  withTiming,
} from "react-native-reanimated";

type Props = {
  color: string;
  variant?: "mission" | "available";
};

const BOX_MISSION = 52;
const BOX_AVAILABLE = 46;

function parseRgb(hex: string): { r: number; g: number; b: number } {
  if (!hex.startsWith("#") || hex.length < 7) {
    return { r: 20, g: 184, b: 166 };
  }
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  };
}

function FleetPulseRing({
  color,
  progress,
  size,
}: {
  color: string;
  progress: SharedValue<number>;
  size: number;
}) {
  const style = useAnimatedStyle(() => ({
    width: size,
    height: size,
    borderRadius: size / 2,
    borderWidth: 2,
    borderColor: color,
    opacity: (1 - progress.value) * 0.55,
    transform: [{ scale: 0.72 + progress.value * 0.85 }],
  }));

  return <Animated.View style={[s.ring, style]} pointerEvents="none" />;
}

/** Anneaux pulsés sous le pin chauffeur (connecté / en mission). */
export function FleetDriverLivePulse({ color, variant = "available" }: Props) {
  const ring1 = useSharedValue(0);
  const ring2 = useSharedValue(0);
  const box = variant === "mission" ? BOX_MISSION : BOX_AVAILABLE;
  const duration = variant === "mission" ? 1500 : 1900;
  const { r, g, b } = parseRgb(color);
  const ringColor = `rgba(${r},${g},${b},0.85)`;
  const coreGlow = `rgba(${r},${g},${b},${variant === "mission" ? 0.28 : 0.2})`;

  useEffect(() => {
    ring1.value = 0;
    ring2.value = 0;
    ring1.value = withRepeat(
      withTiming(1, { duration, easing: Easing.out(Easing.cubic) }),
      -1,
      false
    );
    ring2.value = withDelay(
      Math.round(duration * 0.48),
      withRepeat(
        withTiming(1, { duration, easing: Easing.out(Easing.cubic) }),
        -1,
        false
      )
    );
  }, [duration, ring1, ring2]);

  return (
    <View style={[s.box, { width: box, height: box }]} pointerEvents="none" collapsable={false}>
      <View
        style={[
          s.coreGlow,
          {
            width: box * 0.42,
            height: box * 0.42,
            borderRadius: (box * 0.42) / 2,
            backgroundColor: coreGlow,
          },
        ]}
      />
      <FleetPulseRing color={ringColor} progress={ring1} size={box * 0.55} />
      <FleetPulseRing color={ringColor} progress={ring2} size={box * 0.55} />
    </View>
  );
}

const s = StyleSheet.create({
  box: {
    alignItems: "center",
    justifyContent: "center",
  },
  ring: {
    position: "absolute",
  },
  coreGlow: {
    position: "absolute",
  },
});
