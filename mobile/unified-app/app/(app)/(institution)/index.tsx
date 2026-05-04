import { StyleSheet, Text, View } from "react-native";
import { PermissionGuard } from "../../../src/core/guards";
import { brandSurfaceSoft, ResponsiveContainer, Screen } from "../../../src/design/responsive";

export default function InstitutionHomeScreen() {
  return (
    <PermissionGuard permission="institution:dashboard:read">
      <Screen scroll backgroundColor={brandSurfaceSoft} contentContainerStyle={styles.scroll}>
        <ResponsiveContainer>
          <View style={styles.block}>
            <Text style={styles.title}>Espace institution</Text>
            <Text style={styles.body}>
              Scope V1 fermé : tableau de bord et consultations prioritaires.
            </Text>
          </View>
        </ResponsiveContainer>
      </Screen>
    </PermissionGuard>
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
