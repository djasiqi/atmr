import React, { useState } from "react";
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    ScrollView,
    ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAddressSearch } from "@/hooks/useAddressSearch";
import { AddressSuggestion } from "@/types/enterpriseDispatch";
import { createShadow } from "@/styles/shadowStyles";

const palette = {
    background: "#FFFFFF",
    border: "rgba(15,54,43,0.12)",
    text: "#15362B",
    textSecondary: "#5F7369",
    placeholder: "#91A59D",
    accent: "#0A7F59",
    suggestionBg: "#F5F7F6",
    suggestionBorder: "rgba(15,54,43,0.08)",
    suggestionActive: "rgba(10,127,89,0.08)",
};

interface AddressSelectorProps {
    label: string;
    value: string;
    onChange: (address: string, suggestion?: AddressSuggestion) => void;
    placeholder?: string;
    icon?: keyof typeof Ionicons.glyphMap;
}

export const AddressSelector: React.FC<AddressSelectorProps> = ({
    label,
    value,
    onChange,
    placeholder = "Rechercher une adresse...",
    icon = "location-outline",
}) => {
    const { query, suggestions, loading, search, clear } = useAddressSearch();
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [isFocused, setIsFocused] = useState(false);

    const handleChange = (text: string) => {
        onChange(text);
        if (text.length >= 3) {
            search(text);
            setShowSuggestions(true);
        } else {
            setShowSuggestions(false);
        }
    };

    const handleSelect = (suggestion: AddressSuggestion) => {
        // ✅ Utiliser le label (format complet) si disponible, sinon l'adresse
        const displayAddress = suggestion.label || suggestion.address;
        onChange(displayAddress, suggestion);
        setShowSuggestions(false);
        clear();
    };

    const handleFocus = () => {
        setIsFocused(true);
        if (value.length >= 3) {
            search(value);
            setShowSuggestions(true);
        }
    };

    const handleBlur = () => {
        setIsFocused(false);
        // Délai pour permettre le clic sur une suggestion
        setTimeout(() => setShowSuggestions(false), 200);
    };

    return (
        <View style={styles.container}>
            <Text style={styles.label}>{label}</Text>
            <View style={[styles.inputContainer, isFocused && styles.inputContainerFocused]}>
                <Ionicons name={icon} size={18} color={palette.accent} style={styles.icon} />
                <TextInput
                    style={styles.input}
                    value={value}
                    onChangeText={handleChange}
                    onFocus={handleFocus}
                    onBlur={handleBlur}
                    placeholder={placeholder}
                    placeholderTextColor={palette.placeholder}
                    autoCapitalize="none"
                    autoCorrect={false}
                />
                {loading && (
                    <ActivityIndicator size="small" color={palette.accent} style={styles.loader} />
                )}
                {value.length > 0 && !loading && (
                    <TouchableOpacity
                        onPress={() => {
                            onChange("");
                            clear();
                        }}
                        style={styles.clearButton}
                    >
                        <Ionicons name="close-circle" size={18} color={palette.placeholder} />
                    </TouchableOpacity>
                )}
            </View>

            {showSuggestions && suggestions.length > 0 && (
                <View style={styles.suggestionsContainer}>
                    <ScrollView
                        style={styles.suggestionsList}
                        keyboardShouldPersistTaps="handled"
                        nestedScrollEnabled
                    >
                        {suggestions.map((suggestion, index) => (
                            <TouchableOpacity
                                key={index}
                                style={styles.suggestionItem}
                                onPress={() => handleSelect(suggestion)}
                            >
                                <Ionicons name="location" size={16} color={palette.accent} />
                                <Text style={styles.suggestionText} numberOfLines={2}>
                                    {/* ✅ Afficher le label (format complet) si disponible, sinon l'adresse */}
                                    {suggestion.label || suggestion.address}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                </View>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: 0,
    },
    label: {
        color: palette.text,
        fontSize: 14,
        fontWeight: "600",
        marginBottom: 10,
    },
    inputContainer: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: palette.background,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: palette.border,
        paddingHorizontal: 14,
        paddingVertical: 12,
        gap: 10,
    },
    inputContainerFocused: {
        borderColor: palette.accent,
    },
    icon: {
        marginRight: 4,
    },
    input: {
        flex: 1,
        color: palette.text,
        fontSize: 15,
        padding: 0,
    },
    loader: {
        marginLeft: 8,
    },
    clearButton: {
        padding: 4,
    },
    suggestionsContainer: {
        marginTop: 8,
        maxHeight: 200,
        backgroundColor: palette.background,
        borderRadius: 14,
        borderWidth: 1,
        borderColor: palette.suggestionBorder,
        ...createShadow({
            shadowColor: "rgba(15,54,43,0.1)",
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 1,
            shadowRadius: 8,
            elevation: 4,
        }),
    },
    suggestionsList: {
        maxHeight: 200,
    },
    suggestionItem: {
        flexDirection: "row",
        alignItems: "center",
        paddingHorizontal: 14,
        paddingVertical: 12,
        gap: 10,
        borderBottomWidth: StyleSheet.hairlineWidth,
        borderBottomColor: palette.suggestionBorder,
    },
    suggestionText: {
        flex: 1,
        color: palette.text,
        fontSize: 14,
    },
});

