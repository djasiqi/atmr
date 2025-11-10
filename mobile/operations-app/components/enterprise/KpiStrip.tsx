import React from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  ListRenderItem,
  ViewStyle,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

type KpiItem = {
  id: string;
  label: string;
  value: string | number;
  icon?: keyof typeof ICONS;
  tone?: "default" | "success" | "warning" | "danger";
};

type KpiStripProps = {
  items: KpiItem[];
  style?: ViewStyle;
};

const ICONS: Record<string, string> = {
  inbox: "tray-outline",
  progress: "time-outline",
  assignment: "checkbox-outline",
  incident: "alert-circle-outline",
  driver: "person-outline",
  clock: "alarm-outline",
};

const toneColor = (tone: KpiItem["tone"]) => {
  switch (tone) {
    case "success":
      return "#34D399";
    case "warning":
      return "#FBBF24";
    case "danger":
      return "#F87171";
    default:
      return "#E5EDFF";
  }
};

const renderItem: ListRenderItem<KpiItem> = ({ item }) => (
  <View style={styles.card}>
    {item.icon ? (
      <View style={styles.iconBadge}>
        <Ionicons
          name={(ICONS[item.icon] as any) ?? ICONS.inbox}
          size={16}
          color={toneColor(item.tone)}
        />
      </View>
    ) : null}
    <Text style={[styles.value, { color: toneColor(item.tone) }]}>
      {item.value}
    </Text>
    <Text style={styles.label}>{item.label}</Text>
  </View>
);

export const KpiStrip: React.FC<KpiStripProps> = ({ items, style }) => {
  return (
    <FlatList
      data={items}
      horizontal
      keyExtractor={(item) => item.id}
      renderItem={renderItem}
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={[styles.container, style]}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    gap: 12,
    paddingVertical: 4,
  },
  card: {
    minWidth: 120,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: 16,
    backgroundColor: "rgba(26,40,88,0.85)",
    borderWidth: 1,
    borderColor: "rgba(111,139,255,0.18)",
  },
  iconBadge: {
    alignSelf: "flex-start",
    padding: 6,
    borderRadius: 10,
    backgroundColor: "rgba(20,28,60,0.65)",
    marginBottom: 6,
  },
  value: {
    fontSize: 20,
    fontWeight: "700",
    color: "#FFFFFF",
  },
  label: {
    marginTop: 4,
    fontSize: 12,
    color: "rgba(210,219,255,0.7)",
    letterSpacing: 0.2,
    textTransform: "uppercase",
  },
});

export default KpiStrip;

