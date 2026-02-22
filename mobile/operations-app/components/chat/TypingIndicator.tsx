import React from "react";
import { View, StyleSheet } from "react-native";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withRepeat,
  withSequence,
  withDelay,
  interpolate,
  SharedValue,
} from "react-native-reanimated";

const BRAND = "#00796b";

export default function TypingIndicator() {
  const dot1 = useSharedValue(0);
  const dot2 = useSharedValue(0);
  const dot3 = useSharedValue(0);

  React.useEffect(() => {
    const anim = (sv: SharedValue<number>, delay: number) => {
      sv.value = withDelay(
        delay,
        withRepeat(
          withSequence(withTiming(1, { duration: 300 }), withTiming(0, { duration: 300 })),
          -1,
          false
        )
      );
    };
    anim(dot1, 0);
    anim(dot2, 100);
    anim(dot3, 200);
  }, []);

  const mkStyle = (sv: SharedValue<number>) =>
    useAnimatedStyle(() => ({
      opacity: interpolate(sv.value, [0, 1], [0.3, 1]),
      transform: [{ scale: interpolate(sv.value, [0, 1], [0.7, 1]) }],
    }));

  const s1 = mkStyle(dot1);
  const s2 = mkStyle(dot2);
  const s3 = mkStyle(dot3);

  return (
    <View style={st.container}>
      <View style={st.bubble}>
        <Animated.View style={[st.dot, s1]} />
        <Animated.View style={[st.dot, s2]} />
        <Animated.View style={[st.dot, s3]} />
      </View>
    </View>
  );
}

const st = StyleSheet.create({
  container: { alignSelf: "flex-start", marginLeft: 16, marginVertical: 4 },
  bubble: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: BRAND, marginHorizontal: 2 },
});
