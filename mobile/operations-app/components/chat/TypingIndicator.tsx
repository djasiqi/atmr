// components/chat/TypingIndicator.tsx
// ✅ Animation 3 points pulsants style WhatsApp

import React from "react";
import { View, StyleSheet } from "react-native";
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withTiming,
    withRepeat,
    withSequence,
    interpolate,
    SharedValue,
} from "react-native-reanimated";

export default function TypingIndicator() {
    const dot1 = useSharedValue(0);
    const dot2 = useSharedValue(0);
    const dot3 = useSharedValue(0);

    React.useEffect(() => {
        const animate = (dot: SharedValue<number>, delay: number) => {
            dot.value = withRepeat(
                withSequence(
                    withTiming(1, { duration: 350 }),
                    withTiming(0, { duration: 350 })
                ),
                -1,
                false
            );
        };

        animate(dot1, 0);
        animate(dot2, 150);
        animate(dot3, 300);
    }, []);

    const styleDot = (dot: SharedValue<number>) =>
        useAnimatedStyle(() => ({
            opacity: interpolate(dot.value, [0, 1], [0.3, 1]),
            transform: [
                {
                    scale: interpolate(dot.value, [0, 1], [0.6, 1]),
                },
            ],
        }));

    const dot1Style = styleDot(dot1);
    const dot2Style = styleDot(dot2);
    const dot3Style = styleDot(dot3);

    return (
        <View style={styles.container}>
            <View style={styles.bubble}>
                <Animated.View style={[styles.dot, dot1Style]} />
                <Animated.View style={[styles.dot, dot2Style]} />
                <Animated.View style={[styles.dot, dot3Style]} />
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        alignSelf: "flex-start",
        marginLeft: 16,
        marginVertical: 6,
    },
    bubble: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "#FFFFFF",
        borderRadius: 16,
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderWidth: 1,
        borderColor: "rgba(0,0,0,0.05)",
        shadowColor: "rgba(0,0,0,0.08)",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    dot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: "#0A7F59", // vert Liri
        marginHorizontal: 3,
    },
});

