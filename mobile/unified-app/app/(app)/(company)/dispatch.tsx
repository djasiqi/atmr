import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useRouter } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { PermissionGuard } from "../../../src/core/guards";

const C = {
  pageBg: "#EAF3F1",
  text: "#163A34",
  textMuted: "#5F7369",
  brand: "#0A8F7A",
} as const;

/**
 * Propositions d’assignation (semi-auto). Rempli quand l’API exposera la liste.
 * CTA principal du dashboard : ouvert si le feature flag `company_dispatch_screen_enabled` est actif.
 */
export default function CompanyDispatchProposalsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  return (
    <PermissionGuard permission="company:rides:read">
      <ScrollView
        style={[styles.root, { paddingTop: insets.top }]}
        contentContainerStyle={styles.page}
      >
        <View style={styles.headerRow}>
          <Pressable
            onPress={() => (router.canGoBack() ? router.back() : router.replace("/(app)/(company)/dashboard"))}
            style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.85 }]}
            hitSlop={8}
            accessibilityLabel="Retour"
          >
            <Ionicons name="chevron-back" size={24} color={C.brand} />
          </Pressable>
          <Text style={styles.title}>Propositions</Text>
        </View>
        <Text style={styles.body}>
          Les propositions issues du moteur apparaîtront ici lorsqu’elles seront exposées par l’API.
        </Text>
        <Pressable
          onPress={() => router.push("/(app)/(company)/rides")}
          style={({ pressed }) => [styles.cta, pressed && { opacity: 0.9 }]}
        >
          <Text style={styles.ctaText}>Voir les courses à traiter</Text>
          <Ionicons name="chevron-forward" size={18} color={C.brand} />
        </Pressable>
      </ScrollView>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.pageBg },
  page: { padding: 20, paddingBottom: 40, gap: 12 },
  headerRow: { flexDirection: "row" as const, alignItems: "center" as const, gap: 8, marginBottom: 8 },
  backBtn: { padding: 4 },
  title: { color: C.text, fontSize: 20, fontWeight: "800" as const, flex: 1 },
  body: { color: C.textMuted, fontSize: 15, lineHeight: 22 },
  cta: { flexDirection: "row" as const, alignItems: "center" as const, justifyContent: "space-between" as const, marginTop: 8 },
  ctaText: { color: C.brand, fontSize: 15, fontWeight: "800" as const },
});
