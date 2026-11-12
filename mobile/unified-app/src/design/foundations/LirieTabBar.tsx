import { Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { lirieColor, lirieControl } from "./tokens";

export type LirieTab = {
  key: string;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
};

type Props = {
  tabs: LirieTab[];
  selectedKey: string;
  onSelect: (key: string) => void;
};

/** Barre présentative. Elle ne remplace pas les navigateurs métier. */
export function LirieTabBar({ tabs, selectedKey, onSelect }: Props) {
  return (
    <View accessibilityRole="tablist" style={styles.bar}>
      {tabs.map((tab) => {
        const selected = tab.key === selectedKey;
        return (
          <Pressable
            key={tab.key}
            accessibilityRole="tab"
            accessibilityState={{ selected }}
            onPress={() => onSelect(tab.key)}
            style={styles.tab}
          >
            <Ionicons
              name={tab.icon}
              size={22}
              color={selected ? lirieColor.green : lirieColor.inkMuted}
            />
            <Text
              maxFontSizeMultiplier={2}
              style={[styles.label, selected ? styles.selected : styles.idle]}
            >
              {tab.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    flexWrap: "wrap",
    backgroundColor: lirieColor.white,
    borderTopWidth: 1,
    borderTopColor: lirieColor.border,
  },
  tab: {
    flexGrow: 1,
    flexBasis: "22%",
    minHeight: lirieControl.minHeight,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 6,
    paddingVertical: 8,
  },
  label: {
    fontSize: 13,
    lineHeight: 16,
    fontWeight: "600",
    textAlign: "center",
  },
  selected: { color: lirieColor.green },
  idle: { color: lirieColor.inkMuted },
});
