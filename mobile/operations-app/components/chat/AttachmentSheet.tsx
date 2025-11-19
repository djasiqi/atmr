// components/chat/AttachmentSheet.tsx
// ✅ Action sheet moderne animé style WhatsApp

import React from "react";
import {
    Modal,
    View,
    TouchableOpacity,
    Text,
    StyleSheet,
} from "react-native";
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withTiming,
} from "react-native-reanimated";
import { Ionicons } from "@expo/vector-icons";

type Props = {
    visible: boolean;
    onClose: () => void;
    onPickCamera: () => void;
    onPickGallery: () => void;
    onPickDocument: () => void;
};

export default function AttachmentSheet({
    visible,
    onClose,
    onPickCamera,
    onPickGallery,
    onPickDocument,
}: Props) {
    const slide = useSharedValue(300);
    const opacity = useSharedValue(0);

    React.useEffect(() => {
        if (visible) {
            slide.value = withTiming(0, { duration: 220 });
            opacity.value = withTiming(1, { duration: 180 });
        } else {
            slide.value = withTiming(300, { duration: 200 });
            opacity.value = withTiming(0, { duration: 180 });
        }
    }, [visible]);

    const animStyle = useAnimatedStyle(() => ({
        transform: [{ translateY: slide.value }],
    }));

    const overlayStyle = useAnimatedStyle(() => ({
        opacity: opacity.value,
    }));

    return (
        <Modal visible={visible} transparent animationType="none">
            <Animated.View style={[styles.overlay, overlayStyle]}>
                <TouchableOpacity
                    style={StyleSheet.absoluteFill}
                    onPress={onClose}
                    activeOpacity={1}
                />
            </Animated.View>

            <Animated.View style={[styles.sheet, animStyle]}>
                <Item icon="camera" label="Caméra" onPress={onPickCamera} />
                <Item icon="image" label="Galerie" onPress={onPickGallery} />
                <Item icon="document-text" label="Fichier" onPress={onPickDocument} />

                <TouchableOpacity onPress={onClose} style={styles.cancelBtn}>
                    <Text style={styles.cancelText}>Annuler</Text>
                </TouchableOpacity>
            </Animated.View>
        </Modal>
    );
}

function Item({
    icon,
    label,
    onPress,
}: {
    icon: keyof typeof Ionicons.glyphMap;
    label: string;
    onPress: () => void;
}) {
    return (
        <TouchableOpacity style={styles.item} onPress={onPress}>
            <Ionicons name={icon} color="#0A7F59" size={26} />
            <Text style={styles.label}>{label}</Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.25)",
    },
    sheet: {
        position: "absolute",
        bottom: 0,
        width: "100%",
        backgroundColor: "#FFFFFF",
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        paddingVertical: 20,
        paddingHorizontal: 24,
        elevation: 8,
    },
    item: {
        flexDirection: "row",
        alignItems: "center",
        paddingVertical: 14,
    },
    label: {
        marginLeft: 14,
        fontSize: 17,
        color: "#15362B",
        fontWeight: "600",
    },
    cancelBtn: {
        marginTop: 10,
        paddingVertical: 14,
        borderRadius: 12,
        backgroundColor: "#F5F7F6",
        alignItems: "center",
    },
    cancelText: {
        color: "#0A7F59",
        fontSize: 17,
        fontWeight: "600",
    },
});

