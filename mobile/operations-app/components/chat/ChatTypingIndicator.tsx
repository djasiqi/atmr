// components/chat/ChatTypingIndicator.tsx
import React, { useEffect, useRef } from "react";
import { View, Animated } from "react-native";
import { chatStyles } from "@/styles/chatStyles";

export default function ChatTypingIndicator() {
    const dot1 = useRef(new Animated.Value(0.3)).current;
    const dot2 = useRef(new Animated.Value(0.3)).current;
    const dot3 = useRef(new Animated.Value(0.3)).current;

    useEffect(() => {
        const animateDot = (dot: Animated.Value, delay: number) => {
            return Animated.loop(
                Animated.sequence([
                    Animated.delay(delay),
                    Animated.timing(dot, {
                        toValue: 1,
                        duration: 400,
                        useNativeDriver: true,
                    }),
                    Animated.timing(dot, {
                        toValue: 0.3,
                        duration: 400,
                        useNativeDriver: true,
                    }),
                ])
            );
        };

        const animations = [
            animateDot(dot1, 0),
            animateDot(dot2, 200),
            animateDot(dot3, 400),
        ];

        animations.forEach((anim) => anim.start());

        return () => {
            animations.forEach((anim) => anim.stop());
        };
    }, [dot1, dot2, dot3]);

    return (
        <View style={chatStyles.typingIndicator}>
            <Animated.View
                style={[
                    chatStyles.typingDot,
                    { opacity: dot1 },
                ]}
            />
            <Animated.View
                style={[
                    chatStyles.typingDot,
                    { opacity: dot2 },
                ]}
            />
            <Animated.View
                style={[
                    chatStyles.typingDot,
                    { opacity: dot3 },
                ]}
            />
        </View>
    );
}

