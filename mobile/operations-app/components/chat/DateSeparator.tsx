// components/chat/DateSeparator.tsx
// Séparateur de date pour le chat

import React from "react";
import { View, Text, StyleSheet } from "react-native";

interface Props {
    date: string; // Format ISO: "2025-04-03"
}

export default function DateSeparator({ date }: Props) {
    // Formater la date au format "le 03.04.2025"
    const formatDate = (dateString: string): string => {
        try {
            const dateObj = new Date(dateString);
            const day = dateObj.getDate().toString().padStart(2, "0");
            const month = (dateObj.getMonth() + 1).toString().padStart(2, "0");
            const year = dateObj.getFullYear();
            return `le ${day}.${month}.${year}`;
        } catch (e) {
            return dateString;
        }
    };

    return (
        <View style={styles.container}>
            <View style={styles.line} />
            <Text style={styles.dateText}>{formatDate(date)}</Text>
            <View style={styles.line} />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flexDirection: "row",
        alignItems: "center",
        marginVertical: 16,
        paddingHorizontal: 16,
    },
    line: {
        flex: 1,
        height: 1,
        backgroundColor: "rgba(15,54,43,0.12)",
    },
    dateText: {
        fontSize: 13,
        color: "#5F7369",
        marginHorizontal: 12,
        fontWeight: "500",
    },
});

