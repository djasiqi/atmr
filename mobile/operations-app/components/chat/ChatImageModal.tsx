// components/chat/ChatImageModal.tsx
import React, { useRef } from "react";
import {
    Modal,
    View,
    Image,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    StatusBar,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import {
    PinchGestureHandler,
    GestureHandlerRootView,
    GestureHandlerGestureEvent,
} from "react-native-gesture-handler";
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from "react-native-reanimated";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

interface ChatImageModalProps {
    visible: boolean;
    imageUri: string;
    onClose: () => void;
}

export default function ChatImageModal({
    visible,
    imageUri,
    onClose,
}: ChatImageModalProps) {
    const scale = useSharedValue(1);
    const translateX = useSharedValue(0);
    const translateY = useSharedValue(0);
    const pinchRef = useRef<PinchGestureHandler>(null);

    const pinchHandler = (event: GestureHandlerGestureEvent) => {
        "worklet";
        const nativeEvent = event.nativeEvent as any;
        if (nativeEvent.scale !== undefined) {
            scale.value = nativeEvent.scale;
        }
    };

    const onPinchEnd = () => {
        "worklet";
        if (scale.value < 1) {
            scale.value = withSpring(1);
            translateX.value = withSpring(0);
            translateY.value = withSpring(0);
        } else if (scale.value > 3) {
            scale.value = withSpring(3);
        }
    };

    const animatedStyle = useAnimatedStyle(() => {
        return {
            transform: [
                { translateX: translateX.value },
                { translateY: translateY.value },
                { scale: scale.value },
            ],
        };
    });

    return (
        <Modal
            visible={visible}
            transparent={true}
            animationType="fade"
            onRequestClose={onClose}
            statusBarTranslucent
        >
            <StatusBar hidden />
            <GestureHandlerRootView style={styles.container}>
                <TouchableOpacity
                    style={styles.backdrop}
                    activeOpacity={1}
                    onPress={onClose}
                >
                    <View style={styles.content}>
                        <TouchableOpacity
                            style={styles.closeButton}
                            onPress={onClose}
                            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                        >
                            <Ionicons name="close" size={28} color="#FFFFFF" />
                        </TouchableOpacity>

                        <PinchGestureHandler
                            ref={pinchRef}
                            onGestureEvent={pinchHandler}
                            onEnded={onPinchEnd}
                        >
                            <Animated.View style={[styles.imageContainer, animatedStyle]}>
                                <Image
                                    source={{ uri: imageUri }}
                                    style={styles.image}
                                    resizeMode="contain"
                                />
                            </Animated.View>
                        </PinchGestureHandler>
                    </View>
                </TouchableOpacity>
            </GestureHandlerRootView>
        </Modal>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    backdrop: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.95)",
        justifyContent: "center",
        alignItems: "center",
    },
    content: {
        width: SCREEN_WIDTH,
        height: SCREEN_HEIGHT,
        justifyContent: "center",
        alignItems: "center",
    },
    imageContainer: {
        width: SCREEN_WIDTH,
        height: SCREEN_HEIGHT,
        justifyContent: "center",
        alignItems: "center",
    },
    image: {
        width: SCREEN_WIDTH,
        height: SCREEN_HEIGHT,
    },
    closeButton: {
        position: "absolute",
        top: 50,
        right: 20,
        zIndex: 10,
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: "rgba(10, 127, 89, 0.9)",
        justifyContent: "center",
        alignItems: "center",
    },
});

