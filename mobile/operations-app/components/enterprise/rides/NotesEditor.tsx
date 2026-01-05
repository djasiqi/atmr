import React from "react";
import { View, Text, TextInput, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";

const palette = {
    background: "#FFFFFF",
    border: "rgba(15,54,43,0.12)",
    text: "#15362B",
    textSecondary: "#5F7369",
    placeholder: "#91A59D",
    accent: "#0A7F59",
};

interface NotesEditorProps {
    label: string;
    value: string;
    onChange: (notes: string) => void;
    placeholder?: string;
    maxLength?: number;
}

export const NotesEditor: React.FC<NotesEditorProps> = ({
    label,
    value,
    onChange,
    placeholder = "Ajouter des notes internes...",
    maxLength = 500,
}) => {
    return (
        <View style={styles.container}>
            <View style={styles.labelRow}>
                <Text style={styles.label}>{label}</Text>
                {maxLength && (
                    <Text style={styles.counter}>
                        {value.length}/{maxLength}
                    </Text>
                )}
            </View>
            <View style={styles.inputContainer}>
                <Ionicons name="document-text-outline" size={18} color={palette.accent} style={styles.icon} />
                <TextInput
                    style={styles.input}
                    value={value}
                    onChangeText={onChange}
                    placeholder={placeholder}
                    placeholderTextColor={palette.placeholder}
                    multiline
                    numberOfLines={4}
                    maxLength={maxLength}
                    textAlignVertical="top"
                />
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: 16,
    },
    labelRow: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 8,
    },
    label: {
        color: palette.text,
        fontSize: 14,
        fontWeight: "600",
    },
    counter: {
        color: palette.textSecondary,
        fontSize: 12,
    },
    inputContainer: {
        flexDirection: "row",
        alignItems: "flex-start",
        backgroundColor: palette.background,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: palette.border,
        paddingHorizontal: 14,
        paddingVertical: 12,
        gap: 10,
        minHeight: 100,
    },
    icon: {
        marginTop: 4,
        marginRight: 4,
    },
    input: {
        flex: 1,
        color: palette.text,
        fontSize: 15,
        padding: 0,
        minHeight: 80,
    },
});

