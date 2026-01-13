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
import { useClientSearch } from "@/hooks/useClientSearch";
import { ClientOption } from "@/types/enterpriseDispatch";
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
};

interface ClientSelectorProps {
    label: string;
    value: ClientOption | null;
    onChange: (client: ClientOption | null) => void;
    onNewClient?: () => void;
    placeholder?: string;
}

export const ClientSelector: React.FC<ClientSelectorProps> = ({
    label,
    value,
    onChange,
    onNewClient,
    placeholder = "Rechercher un client...",
}) => {
    const { query, suggestions, loading, search, clear } = useClientSearch();
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [isFocused, setIsFocused] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");

    const handleChange = (text: string) => {
        setSearchQuery(text);
        if (text.length >= 2) {
            search(text);
            setShowSuggestions(true);
        } else {
            setShowSuggestions(false);
            if (text.length === 0) {
                onChange(null);
            }
        }
    };

    const handleSelect = (client: ClientOption) => {
        onChange(client);
        setSearchQuery("");
        setShowSuggestions(false);
        clear();
    };

    const handleClear = () => {
        onChange(null);
        setSearchQuery("");
        clear();
    };

    const handleFocus = () => {
        setIsFocused(true);
        if (searchQuery.length >= 2) {
            search(searchQuery);
            setShowSuggestions(true);
        }
    };

    const handleBlur = () => {
        setIsFocused(false);
        // Délai plus long pour permettre le clic sur "Créer un nouveau client"
        setTimeout(() => setShowSuggestions(false), 300);
    };

    return (
        <View style={styles.container}>
            <Text style={styles.label}>{label}</Text>

            {value ? (
                <View style={styles.selectedContainer}>
                    <View style={styles.selectedInfo}>
                        <Ionicons name="person" size={18} color={palette.accent} />
                        <View style={styles.selectedTextContainer}>
                            <Text style={styles.selectedName}>{value.name}</Text>
                            {value.phone && (
                                <Text style={styles.selectedMeta}>{value.phone}</Text>
                            )}
                        </View>
                    </View>
                    <TouchableOpacity onPress={handleClear} style={styles.clearButton}>
                        <Ionicons name="close-circle" size={20} color={palette.textSecondary} />
                    </TouchableOpacity>
                </View>
            ) : (
                <>
                    <View style={[styles.inputContainer, isFocused && styles.inputContainerFocused]}>
                        <Ionicons name="search-outline" size={18} color={palette.accent} style={styles.icon} />
                        <TextInput
                            style={styles.input}
                            value={searchQuery}
                            onChangeText={handleChange}
                            onFocus={handleFocus}
                            onBlur={handleBlur}
                            placeholder={placeholder}
                            placeholderTextColor={palette.placeholder}
                            autoCapitalize="words"
                            autoCorrect={false}
                        />
                        {loading && (
                            <ActivityIndicator size="small" color={palette.accent} style={styles.loader} />
                        )}
                    </View>

                    {onNewClient &&
                        searchQuery.length >= 2 &&
                        !loading &&
                        suggestions.length === 0 && (
                            <TouchableOpacity 
                                style={styles.newClientButton} 
                                onPress={() => {
                                    setShowSuggestions(false);
                                    onNewClient();
                                }}
                                // Empêcher le blur de masquer le bouton avant le clic
                                onPressIn={() => setShowSuggestions(true)}
                            >
                                <Ionicons name="add-circle-outline" size={18} color={palette.accent} />
                                <Text style={styles.newClientText}>Créer un nouveau client</Text>
                            </TouchableOpacity>
                        )}

                    {showSuggestions && suggestions.length > 0 && (
                        <View style={styles.suggestionsContainer}>
                            <ScrollView
                                style={styles.suggestionsList}
                                keyboardShouldPersistTaps="handled"
                                nestedScrollEnabled
                            >
                                {suggestions.map((client) => (
                                    <TouchableOpacity
                                        key={client.id}
                                        style={styles.suggestionItem}
                                        onPress={() => handleSelect(client)}
                                    >
                                        <Ionicons name="person-circle-outline" size={20} color={palette.accent} />
                                        <View style={styles.suggestionTextContainer}>
                                            <Text style={styles.suggestionName}>{client.name}</Text>
                                            {client.phone && (
                                                <Text style={styles.suggestionMeta}>{client.phone}</Text>
                                            )}
                                        </View>
                                    </TouchableOpacity>
                                ))}
                            </ScrollView>
                        </View>
                    )}
                </>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: 16,
    },
    label: {
        color: palette.text,
        fontSize: 14,
        fontWeight: "600",
        marginBottom: 8,
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
    selectedContainer: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        backgroundColor: palette.background,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: palette.accent,
        paddingHorizontal: 14,
        paddingVertical: 12,
        gap: 10,
    },
    selectedInfo: {
        flexDirection: "row",
        alignItems: "center",
        flex: 1,
        gap: 10,
    },
    selectedTextContainer: {
        flex: 1,
    },
    selectedName: {
        color: palette.text,
        fontSize: 15,
        fontWeight: "600",
    },
    selectedMeta: {
        color: palette.textSecondary,
        fontSize: 13,
        marginTop: 2,
    },
    clearButton: {
        padding: 4,
    },
    newClientButton: {
        flexDirection: "row",
        alignItems: "center",
        marginTop: 8,
        paddingVertical: 10,
        paddingHorizontal: 14,
        gap: 8,
        zIndex: 10, // S'assurer que le bouton est cliquable
        position: "relative",
    },
    newClientText: {
        color: palette.accent,
        fontSize: 14,
        fontWeight: "600",
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
    suggestionTextContainer: {
        flex: 1,
    },
    suggestionName: {
        color: palette.text,
        fontSize: 15,
        fontWeight: "500",
    },
    suggestionMeta: {
        color: palette.textSecondary,
        fontSize: 13,
        marginTop: 2,
    },
});

