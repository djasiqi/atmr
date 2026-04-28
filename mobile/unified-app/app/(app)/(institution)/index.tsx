import { Text, View } from "react-native";
import { PermissionGuard } from "../../../src/core/guards";

export default function InstitutionHomeScreen() {
  return (
    <PermissionGuard permission="institution:dashboard:read">
      <View style={{ flex: 1, justifyContent: "center", padding: 24 }}>
        <Text style={{ fontSize: 22, fontWeight: "700" }}>Espace Institution</Text>
        <Text>Scope V1 ferme: dashboard et consultations prioritaires.</Text>
      </View>
    </PermissionGuard>
  );
}
