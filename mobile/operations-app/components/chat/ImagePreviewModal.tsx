// components/chat/ImagePreviewModal.tsx
// ✅ Pinch-to-zoom + BlurView premium

import React from "react";
import { Modal, StyleSheet, View, TouchableOpacity } from "react-native";
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withTiming,
} from "react-native-reanimated";
import { BlurView } from "expo-blur";
import { Ionicons } from "@expo/vector-icons";
import {
    Gesture,
    GestureDetector,
} from "react-native-gesture-handler";
import { Image } from "react-native";

type Props = {
    visible: boolean;
    uri: string | null;
    onClose: () => void;
};

const AnimatedImage = Animated.createAnimatedComponent(Image);

export default function ImagePreviewModal({
    visible,
    uri,
    onClose,
}: Props) {
    const opacity = useSharedValue(0);
    const scale = useSharedValue(1);
    const translateX = useSharedValue(0);
    const translateY = useSharedValue(0);

    React.useEffect(() => {
        opacity.value = withTiming(visible ? 1 : 0, { duration: 200 });
        if (!visible) {
            // Reset on close
            scale.value = 1;
            translateX.value = 0;
            translateY.value = 0;
        }
    }, [visible]);

    const animatedContainer = useAnimatedStyle(() => ({
        opacity: opacity.value,
    }));

    const animatedImage = useAnimatedStyle(() => ({
        transform: [
            { scale: scale.value },
            { translateX: translateX.value },
            { translateY: translateY.value },
        ],
    }));

    const pinch = Gesture.Pinch()
        .onUpdate((e) => {
            scale.value = e.scale;
        })
        .onEnd(() => {
            if (scale.value < 1) {
                scale.value = withTiming(1, { duration: 200 });
            } else if (scale.value > 3) {
                scale.value = withTiming(3, { duration: 200 });
            }
        });

    const pan = Gesture.Pan()
        .onUpdate((e) => {
            translateX.value = e.translationX;
            translateY.value = e.translationY;
        })
        .onEnd(() => {
            translateX.value = withTiming(0, { duration: 200 });
            translateY.value = withTiming(0, { duration: 200 });
        });

    const composed = Gesture.Simultaneous(pinch, pan);

    if (!uri) return null;

    return (
        <Modal visible={visible} transparent animationType="none">
            <Animated.View style={[styles.container, animatedContainer]}>
                <BlurView intensity={50} tint="dark" style={StyleSheet.absoluteFill} />

                <GestureDetector gesture={composed}>
                    <AnimatedImage
                        source={{ uri }}
                        style={[styles.image, animatedImage]}
                        resizeMode="contain"
                    />
                </GestureDetector>

                <TouchableOpacity style={styles.closeBtn} onPress={onClose}>
                    <Ionicons name="close" size={30} color="#FFFFFF" />
                </TouchableOpacity>
            </Animated.View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: "rgba(0,0,0,0.25)",
        justifyContent: "center",
        alignItems: "center",
    },
    image: {
        width: "100%",
        height: "80%",
    },
    closeBtn: {
        position: "absolute",
        top: 40,
        right: 20,
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: "#0A7F59", // vert Liri
        justifyContent: "center",
        alignItems: "center",
    },
});

