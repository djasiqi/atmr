import { StyleSheet, Text, View } from "react-native";
import { brandSurfaceSoft, ResponsiveContainer, Screen } from "../../src/design/responsive";

export default function UnauthorizedScreen() {
  return (
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <Text style={styles.title}>Accès refusé</Text>
          <Text style={styles.body}>Permission insuffisante pour cette action.</Text>
        </View>
      </ResponsiveContainer>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingVertical: 32,
  },
  block: {
    gap: 12,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: "#163A34",
  },
  body: {
    fontSize: 15,
    lineHeight: 22,
    color: "#5F7369",
  },
});
