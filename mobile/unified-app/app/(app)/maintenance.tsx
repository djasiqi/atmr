import { StyleSheet, View } from "react-native";
import { AppText, brandSurfaceSoft, ResponsiveContainer, Screen } from "../../src/design/responsive";

export default function MaintenanceScreen() {
  return (
    <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
      <ResponsiveContainer>
        <View style={styles.block}>
          <AppText variant="sectionTitle" style={styles.title}>
            Maintenance
          </AppText>
          <AppText variant="bodyMuted" style={styles.body}>
            Service temporairement indisponible.
          </AppText>
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
    color: "#163A34",
  },
  body: {
    lineHeight: 22,
  },
});
