// components/chat/ScrollToBottomButton.tsx
// ✅ Version refactorisée - Bouton flottant animé avec cleanup complet

import React, { useEffect } from "react";
import { TouchableOpacity, StyleSheet } from "react-native";
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withTiming,
    cancelAnimation,
} from "react-native-reanimated";
import { Ionicons } from "@expo/vector-icons";

type Props = {
    visible: boolean;
    onPress: () => void;
    bottomOffset?: number;
};

export default function ScrollToBottomButton({
    visible,
    onPress,
    bottomOffset = 90,
}: Props) {
    const anim = useSharedValue(0);

    useEffect(() => {
        // Annuler toute animation en cours avant de démarrer une nouvelle
        cancelAnimation(anim);
        anim.value = withTiming(visible ? 1 : 0, { duration: 250 });

        // Cleanup : annuler l'animation au démontage
        return () => {
            cancelAnimation(anim);
        };
    }, [visible, anim]);

    const animatedStyle = useAnimatedStyle(() => ({
        opacity: anim.value,
        transform: [{ scale: anim.value }],
    }));

    return (
        <Animated.View
            style={[
                styles.wrapper,
                { bottom: bottomOffset },
                animatedStyle,
            ]}
            pointerEvents={visible ? "auto" : "none"}
        >
            <TouchableOpacity
                style={styles.button}
                onPress={onPress}
                activeOpacity={0.8}
            >
                <Ionicons name="arrow-down" size={22} color="#FFFFFF" />
            </TouchableOpacity>
        </Animated.View>
    );
}

const styles = StyleSheet.create({
    wrapper: {
        position: "absolute",
        right: 20,
        zIndex: 500,
    },
    button: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: "#0A7F59", // vert Liri
        alignItems: "center",
        justifyContent: "center",
        // Ombre premium
        shadowColor: "#0A7F59",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 6,
    },
});
