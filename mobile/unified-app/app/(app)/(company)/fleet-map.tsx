import { useCallback } from "react";
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { useCompanyDriverLiveTracking } from "../../../src/features/company/realtime/useCompanyDriverLiveTracking";
import { EnterpriseDriversMap } from "../../../src/features/company/components/EnterpriseDriversMap";

const C = {
  text: "#163A34",
  textMuted: "#5F7369",
  pageBg: "#EAF3F1",
  brand: "#0A8F7A",
} as const;

export default function CompanyFleetMapScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const live = useCompanyDriverLiveTracking();
  const back = useCallback(() => {
    if (router.canGoBack()) router.back();
    else router.replace("/(app)/(company)/dashboard");
  }, [router]);

  return (
    <PermissionGuard permission="company:dashboard:read">
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <View style={styles.header}>
          <Pressable onPress={back} style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.85 }]} hitSlop={8}>
            <Ionicons name="chevron-back" size={24} color={C.brand} />
          </Pressable>
          <Text style={styles.title}>Carte flotte</Text>
        </View>
        {Platform.OS === "web" ? (
          <View style={styles.webMsg}>
            <Text style={styles.webMsgText}>
              La carte n’est pas disponible sur le web. Utilisez l’application mobile.
            </Text>
          </View>
        ) : (
          <ScrollView contentContainerStyle={styles.page}>
            <EnterpriseDriversMap drivers={live.drivers} showTitleRow />
          </ScrollView>
        )}
      </View>
    </PermissionGuard>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.pageBg },
  header: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    gap: 4,
    paddingHorizontal: 12,
    paddingBottom: 8,
  },
  backBtn: { padding: 4 },
  title: { color: C.text, fontSize: 18, fontWeight: "800" as const, flex: 1 },
  page: { padding: 12, paddingBottom: 32 },
  webMsg: { padding: 20 },
  webMsgText: { color: C.textMuted, fontSize: 14, lineHeight: 20 },
});
