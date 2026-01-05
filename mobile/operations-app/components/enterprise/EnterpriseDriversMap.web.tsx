// components/enterprise/EnterpriseDriversMap.web.tsx
// Version web du composant EnterpriseDriversMap (sans react-native-maps)
import React from "react";
import { View, Text, StyleSheet, ViewStyle } from "react-native";

type DriverMarker = {
    id: string;
    name: string;
    latitude: number;
    longitude: number;
    status?: string;
    eta?: string;
    updatedAt?: string;
};

type EnterpriseDriversMapProps = {
    markers: DriverMarker[];
    style?: ViewStyle;
    fallbackMessage?: string;
};

export const EnterpriseDriversMap: React.FC<EnterpriseDriversMapProps> = ({
    markers,
    style,
    fallbackMessage = "Position des chauffeurs indisponible pour le moment",
}) => {
    return (
        <View style={[styles.container, style]}>
            <View style={styles.overlay}>
                <Text style={styles.overlayTitle}>Carte chauffeurs</Text>
                <Text style={styles.overlayText}>
                    🗺️ Carte non disponible sur le web{"\n"}
                    Utilisez l'application mobile pour voir la carte
                </Text>
                {markers.length > 0 && (
                    <Text style={styles.overlayMeta}>
                        {markers.length} chauffeur{markers.length > 1 ? "s" : ""} en ligne
                    </Text>
                )}
                {markers.length === 0 && (
                    <Text style={styles.overlayMeta}>{fallbackMessage}</Text>
                )}
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        height: 150,
        borderRadius: 22,
        overflow: "hidden",
        marginTop: 10,
        backgroundColor: "#0B1736",
    },
    overlay: {
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(5,11,28,0.72)",
        justifyContent: "center",
        alignItems: "center",
        paddingHorizontal: 24,
        paddingVertical: 16,
    },
    overlayTitle: {
        color: "#FFFFFF",
        fontWeight: "700",
        fontSize: 16,
        marginBottom: 6,
    },
    overlayText: {
        color: "rgba(214,224,255,0.85)",
        textAlign: "center",
        fontSize: 13,
    },
    overlayMeta: {
        color: "rgba(148,163,255,0.7)",
        marginTop: 10,
        fontSize: 11,
        textTransform: "uppercase",
        letterSpacing: 0.4,
    },
});

