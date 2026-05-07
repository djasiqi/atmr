import { useCallback } from "react";
import { Pressable, ScrollView, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { useCompanyDriverLiveTracking } from "../../../src/features/company/realtime/useCompanyDriverLiveTracking";
import { EnterpriseDriversMap } from "../../../src/features/company/components/EnterpriseDriversMap";
import { AppText, Screen } from "../../../src/design/responsive";

const C = {
  text: "#163A34",
  pageBg: "#EAF3F1",
  brand: "#0A8F7A",
} as const;

export default function CompanyFleetMapScreen() {
  const router = useRouter();
  const live = useCompanyDriverLiveTracking();
  const back = useCallback(() => {
    if (router.canGoBack()) router.back();
    else router.replace("/(app)/(company)/dashboard");
  }, [router]);

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen backgroundColor={C.pageBg} withHorizontalPadding={false} scroll={false}>
        <View style={styles.fill}>
          <View style={styles.header}>
            <Pressable onPress={back} style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.85 }]} hitSlop={8}>
              <Ionicons name="chevron-back" size={24} color={C.brand} />
            </Pressable>
            <AppText variant="sectionTitle" style={styles.title}>
              Carte flotte
            </AppText>
          </View>
          {/* ScrollView interne : évite d’imbriquer deux <Screen> (double safe area). Le Screen externe fournit déjà les insets. */}
          <ScrollView contentContainerStyle={styles.page} style={styles.scroll}>
            <EnterpriseDriversMap drivers={live.drivers} />
          </ScrollView>
        </View>
      </Screen>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  scroll: { flex: 1 },
  header: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 4,
    paddingHorizontal: 12,
    paddingBottom: 8,
  },
  backBtn: { padding: 4 },
  title: { color: C.text, fontWeight: "800" as const, flex: 1 },
  page: { padding: 12, paddingBottom: 32 },
});
