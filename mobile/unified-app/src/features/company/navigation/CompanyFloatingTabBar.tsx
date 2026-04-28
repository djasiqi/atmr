import { useState } from "react";
import { Modal, Platform, Pressable, Text, useWindowDimensions, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import type { BottomTabBarProps } from "@react-navigation/bottom-tabs";
import { useRouter, type Href } from "expo-router";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { useSession } from "../../../core/sessionProvider";
import { useCompanyChatUnread } from "../hooks";

const C = {
  border: "rgba(228, 231, 236, 0.9)",
  bg: "#FFFCFC",
  text: "#2D3748",
  textMuted: "#7A808A",
  brand: "#0A8F7A",
} as const;

const HIDDEN_SHEET_ROUTES: { name: "clients" | "settings"; label: string; icon: keyof typeof Ionicons.glyphMap; href: Href }[] = [
  { name: "clients", label: "Clients", icon: "people-outline", href: "/(app)/(company)/clients" as Href },
  { name: "settings", label: "Paramètres", icon: "settings-outline", href: "/(app)/(company)/settings" as Href },
];

function useCanCreateCompanyRide(): boolean {
  const { can, activeContext } = useSession();
  const roleGuardsEnabled = isFeatureEnabled("company_mobile_role_guards_enabled");
  const contextPermissions = activeContext?.permissions ?? [];
  if (!roleGuardsEnabled) return true;
  if (contextPermissions.includes("company:rides:create")) return can("company:rides:create");
  return can("company:rides:read");
}

function useActiveRouteName(state: BottomTabBarProps["state"]): string {
  if (!state?.routes?.length) return "";
  return state.routes[state.index]?.name ?? "";
}

/**
 * Barre d’onglets flottante (pilule) avec CTA centrale « nouvelle course »
 * (patron 2 + bouton rond + 2). « Autres » ouvre Clients et Paramètres.
 */
export function CompanyFloatingTabBar({ state, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const router = useRouter();
  const canCreateRide = useCanCreateCompanyRide();
  const [moreOpen, setMoreOpen] = useState(false);
  const { unreadCount: chatUnread } = useCompanyChatUnread();
  const current = useActiveRouteName(state);
  const maxBarWidth = Math.min(480, width - 32);
  const focusedFromSheet = HIDDEN_SHEET_ROUTES.some((x) => x.name === current);
  const bottomPad = Math.max(12, insets.bottom + 4);
  const totalBarAreaHeight = 64 + insets.bottom;

  return (
    <>
    <View
      style={{ height: totalBarAreaHeight, backgroundColor: "transparent" }}
      pointerEvents="box-none"
    >
      <View
        pointerEvents="box-none"
        style={{
          position: "absolute" as const,
          left: 0,
          right: 0,
          bottom: 0,
          alignItems: "center" as const,
          paddingBottom: bottomPad,
        }}
      >
        <View
          style={[
            {
              maxWidth: maxBarWidth,
              width: "100%" as const,
              marginHorizontal: 16,
              minHeight: 56,
              flexDirection: "row" as const,
              alignItems: "center" as const,
              paddingHorizontal: 4,
              paddingVertical: 4,
              backgroundColor: C.bg,
              borderWidth: 1,
              borderColor: C.border,
              borderRadius: 9999,
            },
            Platform.select({
              web: { boxShadow: "0 4px 20px rgba(15, 23, 42, 0.08)" },
              default: {
                shadowColor: "#0F172A",
                shadowOpacity: 0.1,
                shadowOffset: { width: 0, height: 4 },
                shadowRadius: 12,
                elevation: 6,
              },
            }),
          ]}
        >
          <BarTabButton
            label="Dashboard"
            icon="speedometer-outline"
            active={current === "dashboard" && !focusedFromSheet}
            onPress={() => {
              setMoreOpen(false);
              navigation.navigate("dashboard" as never);
            }}
          />
          <BarTabButton
            label="Courses"
            icon="car-outline"
            active={current === "rides" && !focusedFromSheet}
            onPress={() => {
              setMoreOpen(false);
              navigation.navigate("rides" as never);
            }}
          />
          <View
            style={{
              alignItems: "center" as const,
              justifyContent: "center" as const,
              width: 52,
            }}
          >
            <Pressable
              onPress={() => {
                setMoreOpen(false);
                if (canCreateRide) {
                  void router.push({ pathname: "/(app)/(company)/rides", params: { create: "1" } } as Href);
                }
              }}
              disabled={!canCreateRide}
              accessibilityLabel="Nouvelle course"
              style={({ pressed }) => [
                {
                  width: 44,
                  height: 44,
                  borderRadius: 22,
                  backgroundColor: canCreateRide ? C.brand : "rgba(10, 143, 122, 0.35)",
                  alignItems: "center" as const,
                  justifyContent: "center" as const,
                },
                Platform.select({
                  web: { boxShadow: "0 1px 4px rgba(10, 58, 52, 0.2)" } as const,
                  default: {
                    elevation: 2,
                    shadowColor: "#163A34",
                    shadowOpacity: 0.2,
                    shadowOffset: { width: 0, height: 1 },
                    shadowRadius: 2,
                  },
                }),
                pressed && canCreateRide && { opacity: 0.92 },
              ]}
            >
              <Ionicons name="add" size={26} color="#FFFFFF" />
            </Pressable>
          </View>
          <BarTabButton
            label="Chat"
            icon="chatbubble-ellipses-outline"
            active={current === "chat" && !focusedFromSheet}
            badgeCount={chatUnread}
            onPress={() => {
              setMoreOpen(false);
              navigation.navigate("chat" as never);
            }}
          />
          <BarTabButton
            label="Autres"
            icon="grid-outline"
            active={moreOpen || focusedFromSheet}
            onPress={() => setMoreOpen((v) => !v)}
          />
        </View>
      </View>
    </View>

    <Modal visible={moreOpen} animationType="fade" transparent onRequestClose={() => setMoreOpen(false)}>
        <View style={{ flex: 1, justifyContent: "flex-end" as const, backgroundColor: "rgba(0,0,0,0.4)" }}>
          <Pressable
            onPress={() => setMoreOpen(false)}
            style={{ position: "absolute" as const, top: 0, left: 0, right: 0, bottom: 0 }}
            accessibilityLabel="Fermer le menu"
          />
          <View
            style={[
              {
                backgroundColor: "#FFF",
                borderTopLeftRadius: 16,
                borderTopRightRadius: 16,
                padding: 20,
                paddingBottom: insets.bottom + 20,
                borderWidth: 1,
                borderColor: C.border,
                gap: 8,
                maxWidth: 480,
                width: "100%" as const,
                alignSelf: "center" as const,
              },
            ]}
          >
            <Text style={{ fontSize: 16, fontWeight: "700", color: C.text, marginBottom: 4 }}>Autres écrans</Text>
            {HIDDEN_SHEET_ROUTES.map((row) => {
              const active = current === row.name;
              return (
                <Pressable
                  key={row.name}
                  onPress={() => {
                    setMoreOpen(false);
                    void router.push(row.href);
                  }}
                  style={({ pressed }) => [
                    {
                      flexDirection: "row" as const,
                      alignItems: "center" as const,
                      gap: 10,
                      paddingVertical: 12,
                      paddingHorizontal: 10,
                      borderRadius: 12,
                      backgroundColor: active ? "rgba(10, 143, 122, 0.1)" : "transparent",
                      borderWidth: 1,
                      borderColor: active ? C.brand : C.border,
                    },
                    pressed && { opacity: 0.9 },
                  ]}
                >
                  <Ionicons name={row.icon} size={22} color={active ? C.brand : C.textMuted} />
                  <Text style={{ fontSize: 15, fontWeight: "600", color: active ? C.brand : C.text }}>{row.label}</Text>
                  <View style={{ flex: 1 }} />
                  <Ionicons name="chevron-forward" size={18} color={C.textMuted} />
                </Pressable>
              );
            })}
          </View>
        </View>
    </Modal>
    </>
  );
}

function BarTabButton({
  label,
  icon,
  active,
  onPress,
  badgeCount = 0,
}: {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  active: boolean;
  onPress: () => void;
  /** Nombre de notifications (ex. messages reçus non lus). */
  badgeCount?: number;
}) {
  const a11y =
    badgeCount > 0
      ? `${label}, ${badgeCount} non lu${badgeCount > 1 ? "s" : ""}`
      : label;
  return (
    <Pressable
      onPress={onPress}
      accessibilityLabel={a11y}
      style={({ pressed }) => [
        { flex: 1, alignItems: "center", justifyContent: "center", paddingVertical: 4, borderRadius: 24 },
        pressed && { backgroundColor: "rgba(0,0,0,0.04)" },
      ]}
    >
      <View style={{ position: "relative", width: 28, height: 24, alignItems: "center", justifyContent: "center" }}>
        <Ionicons name={icon} size={22} color={active ? C.brand : C.textMuted} />
        {badgeCount > 0 ? (
          <View
            style={{
              position: "absolute" as const,
              right: -2,
              top: -2,
              minWidth: 16,
              minHeight: 16,
              paddingHorizontal: 4,
              borderRadius: 8,
              backgroundColor: "rgba(10, 143, 122, 0.2)",
              borderWidth: 1,
              borderColor: C.brand,
              alignItems: "center" as const,
              justifyContent: "center" as const,
            }}
            accessibilityLabel={`${badgeCount}`}
            accessibilityRole="text"
            accessible
          >
            <Text
              numberOfLines={1}
              style={{ fontSize: 9, fontWeight: "800" as const, color: C.brand, lineHeight: 10 }}
            >
              {badgeCount > 99 ? "99+" : String(badgeCount)}
            </Text>
          </View>
        ) : null}
      </View>
      <Text
        numberOfLines={1}
        style={{
          marginTop: 2,
          fontSize: 9,
          fontWeight: "700" as const,
          color: active ? C.brand : C.textMuted,
        }}
      >
        {label}
      </Text>
    </Pressable>
  );
}
