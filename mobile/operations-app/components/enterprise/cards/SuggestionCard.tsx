import React from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ViewStyle,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { EnterpriseCard } from "./EnterpriseCard";

type DriverSuggestion = {
  id: string | number;
  name: string;
  eta: string;
  distance?: string;
  score?: number;
  tags?: string[];
};

type SuggestionCardProps = {
  bookingLabel: string;
  address: string;
  scheduledFor: string;
  suggestions: DriverSuggestion[];
  onAccept?: (driverId: string | number) => void;
  onDecline?: () => void;
  onOpenDetails?: () => void;
  style?: ViewStyle;
};

export const SuggestionCard: React.FC<SuggestionCardProps> = ({
  bookingLabel,
  address,
  scheduledFor,
  suggestions,
  onAccept,
  onDecline,
  onOpenDetails,
  style,
}) => {
  const top = suggestions.slice(0, 3);

  return (
    <EnterpriseCard style={[styles.card, style]}>
      <View style={styles.header}>
        <Text style={styles.booking}>{bookingLabel}</Text>
        <TouchableOpacity onPress={onOpenDetails}>
          <Ionicons name="chevron-forward" size={18} color="#93C5FD" />
        </TouchableOpacity>
      </View>

      <Text style={styles.address} numberOfLines={2}>
        {address}
      </Text>
      <Text style={styles.schedule}>Pickup prévu : {scheduledFor}</Text>

      <View style={styles.list}>
        {top.map((driver, index) => (
          <View key={driver.id} style={styles.suggestionRow}>
            <View style={styles.rankBadge}>
              <Text style={styles.rankText}>{index + 1}</Text>
            </View>
            <View style={styles.driverContent}>
              <Text style={styles.driverName}>{driver.name}</Text>
              <View style={styles.metaRow}>
                <Ionicons name="navigate-outline" size={14} color="#69C9FF" />
                <Text style={styles.metaText}>{driver.eta}</Text>
                {driver.distance ? (
                  <Text style={styles.metaText}>• {driver.distance}</Text>
                ) : null}
              </View>
              {driver.tags?.length ? (
                <View style={styles.tagRow}>
                  {driver.tags.map((tag) => (
                    <View key={tag} style={styles.tag}>
                      <Text style={styles.tagText}>{tag}</Text>
                    </View>
                  ))}
                </View>
              ) : null}
            </View>
            {onAccept ? (
              <TouchableOpacity
                style={styles.acceptButton}
                onPress={() => onAccept(driver.id)}
              >
                <Ionicons name="checkmark" size={18} color="#0B1736" />
              </TouchableOpacity>
            ) : null}
          </View>
        ))}
      </View>

      <View style={styles.actions}>
        {onDecline ? (
          <TouchableOpacity style={styles.secondaryButton} onPress={onDecline}>
            <Text style={styles.secondaryText}>Refuser</Text>
          </TouchableOpacity>
        ) : null}
        <TouchableOpacity
          style={[styles.primaryButton, !onAccept && styles.primaryButtonWide]}
          onPress={onOpenDetails}
        >
          <Text style={styles.primaryText}>Modifier</Text>
        </TouchableOpacity>
      </View>
    </EnterpriseCard>
  );
};

const styles = StyleSheet.create({
  card: {
    gap: 10,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  booking: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 15,
  },
  address: {
    color: "rgba(214,224,255,0.9)",
    fontSize: 13,
  },
  schedule: {
    color: "rgba(141,160,220,0.85)",
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  list: {
    gap: 12,
    marginTop: 8,
  },
  suggestionRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
  },
  rankBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "rgba(73,109,241,0.35)",
    justifyContent: "center",
    alignItems: "center",
  },
  rankText: {
    color: "#FFFFFF",
    fontWeight: "700",
  },
  driverContent: {
    flex: 1,
    gap: 4,
  },
  driverName: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 15,
  },
  metaRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  metaText: {
    color: "rgba(182,196,245,0.9)",
    fontSize: 12,
  },
  tagRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  tag: {
    backgroundColor: "rgba(94,234,212,0.18)",
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  tagText: {
    color: "#5EEAD4",
    fontSize: 11,
    fontWeight: "600",
  },
  acceptButton: {
    backgroundColor: "#5EEAD4",
    borderRadius: 14,
    padding: 8,
  },
  actions: {
    flexDirection: "row",
    gap: 12,
    marginTop: 8,
  },
  secondaryButton: {
    flex: 1,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(148,163,255,0.4)",
    paddingVertical: 12,
    alignItems: "center",
  },
  secondaryText: {
    color: "rgba(208,217,255,0.9)",
    fontWeight: "600",
  },
  primaryButton: {
    flex: 1,
    borderRadius: 14,
    backgroundColor: "#60A5FA",
    paddingVertical: 12,
    alignItems: "center",
  },
  primaryButtonWide: {
    flex: 2,
  },
  primaryText: {
    color: "#0B1736",
    fontWeight: "700",
  },
});

export default SuggestionCard;

