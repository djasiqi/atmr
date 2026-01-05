// components/dashboard/MissionGroupHeader.tsx
import React from "react";
import { View, Text } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { styles } from "@/styles/missionGroupStyles";

type Props = {
    location: string;
    count: number;
    type: "pickup" | "dropoff";
};

const MissionGroupHeader: React.FC<Props> = ({ location, count, type }) => {
    const icon = type === "pickup" ? "location" : "flag";
    const label = type === "pickup" ? "MÊME POINT DE DÉPART" : "MÊME DESTINATION";

    return (
        <View style={styles.groupHeaderContainer}>
            <View style={styles.groupHeaderContent}>
                <Ionicons
                    name={icon}
                    size={20}
                    color="#FF6B35"
                    style={styles.groupHeaderIcon}
                />
                <View style={styles.groupHeaderTextContainer}>
                    <Text style={styles.groupHeaderLabel}>
                        ⚠️ {label} ({count} {count === 1 ? "mission" : "missions"})
                    </Text>
                    <Text style={styles.groupHeaderLocation}>{location}</Text>
                </View>
            </View>
        </View>
    );
};

export default MissionGroupHeader;

