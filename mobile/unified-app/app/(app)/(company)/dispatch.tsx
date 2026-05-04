import { Pressable, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { PermissionGuard } from "../../../src/core/guards";
import { AppText, Screen, useAppViewport, useResponsiveTokens } from "../../../src/design/responsive";

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
  const router = useRouter();
  const { horizontalPadding } = useAppViewport();
  const t = useResponsiveTokens();
  return (
    <PermissionGuard permission="company:rides:read">
      <Screen
        scroll
        backgroundColor={C.pageBg}
        withHorizontalPadding={false}
        contentContainerStyle={[styles.page, { paddingHorizontal: horizontalPadding, gap: t.spacingSm }]}
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
          <AppText variant="sectionTitle" style={styles.title}>
            Propositions
          </AppText>
        </View>
        <AppText variant="bodyMuted" style={styles.body}>
          Les propositions issues du moteur apparaîtront ici lorsqu’elles seront exposées par l’API.
        </AppText>
        <Pressable
          onPress={() => router.push("/(app)/(company)/rides")}
          style={({ pressed }) => [styles.cta, pressed && { opacity: 0.9 }]}
        >
          <AppText variant="label" style={styles.ctaText}>
            Voir les courses à traiter
          </AppText>
          <Ionicons name="chevron-forward" size={18} color={C.brand} />
        </Pressable>
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  page: { paddingTop: 20, paddingBottom: 40 },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  backBtn: { padding: 4 },
  title: { color: C.text, flex: 1 },
  body: { lineHeight: 22 },
  cta: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    alignSelf: "flex-start",
    marginTop: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "rgba(10,143,122,0.12)",
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.35)",
  },
  ctaText: { color: C.brand },
});
