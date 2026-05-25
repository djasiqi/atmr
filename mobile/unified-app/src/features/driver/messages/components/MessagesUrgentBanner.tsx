import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../../design/ui/AppText";

type Props = {
  count: number;
  onPress: () => void;
};

export function MessagesUrgentBanner({ count, onPress }: Props) {
  if (count <= 0) return null;

  return (
    <Pressable
      style={({ pressed }) => [styles.banner, pressed && styles.bannerPressed]}
      onPress={onPress}
      accessibilityRole="button"
    >
      <Ionicons name="alert-circle" size={20} color="#B45309" />
      <View style={styles.text}>
        <AppText variant="body" style={styles.title}>
          {count} conversation{count > 1 ? "s" : ""} urgente{count > 1 ? "s" : ""}
        </AppText>
        <AppText variant="caption" style={styles.sub}>
          Appuyez pour filtrer
        </AppText>
      </View>
      <Ionicons name="chevron-forward" size={18} color="#B45309" />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginHorizontal: 16,
    marginVertical: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: "#FEF3C7",
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "#FCD34D",
  },
  bannerPressed: { opacity: 0.92 },
  text: { flex: 1 },
  title: { fontWeight: "600", color: "#92400E" },
  sub: { color: "#B45309", marginTop: 2 },
});
