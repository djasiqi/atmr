import React, { useEffect } from "react";
import { TouchableOpacity, StyleSheet, Platform } from "react-native";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  cancelAnimation,
} from "react-native-reanimated";
import { Ionicons } from "@expo/vector-icons";

const BRAND = "#00796b";

type Props = {
  visible: boolean;
  onPress: () => void;
  bottomOffset?: number;
};

export default function ScrollToBottomButton({ visible, onPress, bottomOffset = 90 }: Props) {
  const anim = useSharedValue(0);

  useEffect(() => {
    cancelAnimation(anim);
    anim.value = withTiming(visible ? 1 : 0, { duration: 200 });
    return () => { cancelAnimation(anim); };
  }, [visible, anim]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: anim.value,
    transform: [{ scale: anim.value }],
  }));

  return (
    <Animated.View
      style={[st.wrapper, { bottom: bottomOffset, pointerEvents: visible ? "auto" : "none" }, animatedStyle]}
    >
      <TouchableOpacity style={st.btn} onPress={onPress} activeOpacity={0.8}>
        <Ionicons name="chevron-down" size={18} color={BRAND} />
      </TouchableOpacity>
    </Animated.View>
  );
}

const shadow = Platform.OS === "web"
  ? { boxShadow: "0 2px 8px rgba(0,0,0,0.12)" }
  : { shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.12, shadowRadius: 8, elevation: 4 };

const st = StyleSheet.create({
  wrapper: { position: "absolute", right: 16, zIndex: 500 },
  btn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "#e5e7eb",
    ...shadow,
  },
});
