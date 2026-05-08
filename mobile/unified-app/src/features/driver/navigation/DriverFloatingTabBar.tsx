import { useMemo, useState } from "react";
import { Modal, Platform, Pressable, StyleSheet, View } from "react-native";
import {
  BaseFloatingBar,
  computeCompanyFloatingBottomPad,
} from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { useAppViewport } from "../../../design/responsive/useAppViewport";
import { Ionicons } from "@expo/vector-icons";
import type { BottomTabBarProps } from "@react-navigation/bottom-tabs";
import { useRouter, useSegments, type Href } from "expo-router";
import { useDriverChatUnreadCount } from "../chatHooks";

const C = {
  border: "rgba(228, 231, 236, 0.9)",
  text: "#2D3748",
  textMuted: "#7A808A",
  brand: "#0A8F7A",
} as const;

/** Padding scroll pour que le dernier contenu ne passe pas sous la pilule (company dashboard ≈ 80). */
export const DRIVER_FLOATING_TAB_SCROLL_PADDING = 96;

const MORE_SHEET_ROUTES: {
  name: "schedule" | "profile";
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  href: Href;
}[] = [
  {
    name: "schedule",
    label: "Planning",
    icon: "calendar-outline",
    href: "/(app)/(driver)/schedule",
  },
  {
    name: "profile",
    label: "Profil chauffeur",
    icon: "person-outline",
    href: "/(app)/(driver)/profile",
  },
];

/** Web / mobile web : pas de contour rectangulaire au focus. */
const PRESSABLE_WEB_SUPPRESS_SQUARE_HALO = Platform.select({
  web: {
    cursor: "pointer",
    outlineWidth: 0,
    outlineStyle: "none",
    // @ts-expect-error RN web
    WebkitTapHighlightColor: "transparent",
  } as const,
  default: undefined,
});

function useDriverTabHighlight(): {
  home: boolean;
  trips: boolean;
  chat: boolean;
  moreRoute: "schedule" | "profile" | null;
} {
  const segments = useSegments();
  return useMemo(() => {
    const last = segments.at(-1) ?? "";
    const prev = segments.at(-2) ?? "";

    const home =
      (segments.length === 2 && segments[0] === "(app)" && segments[1] === "(driver)") ||
      (last === "index" && prev === "(driver)");

    const trips = last === "trips" || prev === "trips";
    const chat = last === "chat";
    const moreRoute =
      last === "schedule" ? "schedule" : last === "profile" ? "profile" : null;

    return { home, trips, chat, moreRoute };
  }, [segments]);
}

/**
 * Barre d’onglets flottante alignée sur {@link CompanyFloatingTabBar} :
 * Accueil, Courses, FAB (liste missions), Chat, Autres (Planning + Profil).
 */
export function DriverFloatingTabBar({ navigation }: BottomTabBarProps) {
  const { usableWidth, bottomInset, horizontalPadding } = useAppViewport();
  const { isLargeText } = useAccessibilityScale();
  const router = useRouter();
  const [moreOpen, setMoreOpen] = useState(false);
  const chatUnread = useDriverChatUnreadCount();
  const tab = useDriverTabHighlight();
  const maxBarWidth = Math.min(480, usableWidth - 2 * horizontalPadding);
  const bottomPad = computeCompanyFloatingBottomPad(bottomInset);
  const totalBarAreaHeight = 64 + bottomInset;
  const focusedFromSheet = tab.moreRoute != null;

  return (
    <>
      <BaseFloatingBar
        containerHeight={totalBarAreaHeight}
        paddingBottom={bottomPad}
        maxBarWidth={maxBarWidth}
        horizontalPadding={horizontalPadding}
        preset="company"
        minInnerHeight={56}
        minInnerHeightLargeText={62}
        isLargeText={isLargeText}
      >
        <BarTabButton
          label="Accueil"
          icon="speedometer-outline"
          active={tab.home && !focusedFromSheet}
          onPress={() => {
            setMoreOpen(false);
            navigation.navigate("index" as never);
          }}
        />
        <BarTabButton
          label="Courses"
          icon="car-outline"
          active={tab.trips && !focusedFromSheet}
          onPress={() => {
            setMoreOpen(false);
            navigation.navigate("trips" as never);
          }}
        />
        <View style={styles.fabSlot}>
          <Pressable
            onPress={() => {
              setMoreOpen(false);
              navigation.navigate("missions" as never);
            }}
            accessibilityLabel="Missions"
            accessibilityRole="button"
            android_ripple={
              Platform.OS === "android"
                ? { color: "rgba(255, 255, 255, 0.35)", borderless: true }
                : undefined
            }
            style={({ pressed }) => [
              styles.fabOuter,
              pressed && styles.fabOuterPressed,
              Platform.OS === "web" ? PRESSABLE_WEB_SUPPRESS_SQUARE_HALO : null,
            ]}
          >
            <Ionicons name="add" size={26} color="#FFFFFF" />
          </Pressable>
        </View>
        <BarTabButton
          label="Chat"
          icon="chatbubble-ellipses-outline"
          active={tab.chat && !focusedFromSheet}
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
      </BaseFloatingBar>

      <Modal visible={moreOpen} animationType="slide" transparent onRequestClose={() => setMoreOpen(false)}>
        <View style={styles.modalBackdrop}>
          <Pressable
            onPress={() => setMoreOpen(false)}
            style={{ position: "absolute" as const, top: 0, left: 0, right: 0, bottom: 0 }}
            accessibilityLabel="Fermer le menu"
          />
          <View style={[styles.sheetPanel, { paddingBottom: bottomInset + 20 }]}>
            <AppText variant="sectionTitle" style={styles.sheetTitle}>
              Autres écrans
            </AppText>
            {MORE_SHEET_ROUTES.map((row) => {
              const active = tab.moreRoute === row.name;
              return (
                <Pressable
                  key={row.name}
                  onPress={() => {
                    setMoreOpen(false);
                    void router.push(row.href);
                  }}
                  style={({ pressed }) => [
                    styles.sheetRow,
                    active ? styles.sheetRowActive : styles.sheetRowIdle,
                    pressed && styles.sheetRowPressed,
                  ]}
                >
                  <Ionicons name={row.icon} size={22} color={active ? C.brand : C.textMuted} />
                  <AppText
                    variant="body"
                    maxFontSizeMultiplier={1.35}
                    style={[styles.sheetRowLabel, { color: active ? C.brand : C.text }]}
                  >
                    {row.label}
                  </AppText>
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
  badgeCount?: number;
}) {
  const a11y =
    badgeCount > 0
      ? `${label}, ${badgeCount} non lu${badgeCount > 1 ? "s" : ""}`
      : label;
  return (
    <View style={styles.tabHit}>
      <Pressable
        onPress={onPress}
        accessibilityLabel={a11y}
        accessibilityRole="tab"
        accessibilityState={{ selected: active }}
        android_ripple={
          Platform.OS === "android"
            ? { color: "rgba(10, 58, 52, 0.14)", borderless: false, foreground: true }
            : undefined
        }
        style={({ pressed }) => [
          styles.tabPressable,
          active && styles.tabIconShellActive,
          pressed && styles.tabHitPressed,
          Platform.OS === "web" ? PRESSABLE_WEB_SUPPRESS_SQUARE_HALO : null,
        ]}
      >
        <View style={styles.iconBadgeWrap}>
          <Ionicons name={icon} size={22} color={active ? C.brand : C.textMuted} />
          {badgeCount > 0 ? (
            <View style={styles.badge} accessibilityLabel={`${badgeCount}`} accessibilityRole="text" accessible>
              <AppText
                variant="caption"
                numberOfLines={1}
                maxFontSizeMultiplier={1.25}
                style={styles.badgeText}
              >
                {badgeCount > 99 ? "99+" : String(badgeCount)}
              </AppText>
            </View>
          ) : null}
        </View>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  tabHit: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 48,
  },
  tabHitPressed: {
    opacity: Platform.OS === "ios" ? 0.88 : 1,
  },
  tabPressable: {
    paddingHorizontal: 13,
    paddingVertical: 9,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 44,
    minHeight: 44,
    overflow: "hidden",
    alignSelf: "center",
  },
  tabIconShellActive: {
    backgroundColor: "rgba(10, 143, 122, 0.1)",
  },
  iconBadgeWrap: {
    position: "relative",
    width: 28,
    height: 24,
    alignItems: "center",
    justifyContent: "center",
  },
  badge: {
    position: "absolute",
    right: -2,
    top: -2,
    minWidth: 16,
    minHeight: 16,
    paddingHorizontal: 4,
    borderRadius: 8,
    backgroundColor: "rgba(10, 143, 122, 0.2)",
    borderWidth: 1,
    borderColor: C.brand,
    alignItems: "center",
    justifyContent: "center",
  },
  badgeText: {
    fontWeight: "800",
    color: C.brand,
    fontSize: 10,
    lineHeight: 12,
  },
  fabSlot: {
    alignItems: "center",
    justifyContent: "center",
    width: 52,
  },
  fabOuter: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.brand,
    alignItems: "center",
    justifyContent: "center",
    ...Platform.select({
      web: {
        boxShadow: "0 1px 4px rgba(10, 58, 52, 0.2)",
      } as const,
      default: {
        elevation: 2,
        shadowColor: "#163A34",
        shadowOpacity: 0.2,
        shadowOffset: { width: 0, height: 1 },
        shadowRadius: 2,
      },
    }),
  },
  fabOuterPressed: {
    opacity: 0.9,
  },
  modalBackdrop: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(15, 23, 42, 0.38)",
  },
  sheetPanel: {
    backgroundColor: "#FFFFFF",
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    paddingHorizontal: 20,
    paddingTop: 18,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: C.border,
    gap: 6,
    maxWidth: 480,
    width: "100%",
    alignSelf: "center",
    ...Platform.select({
      web: {
        boxShadow: "0 -8px 40px rgba(15, 23, 42, 0.08)",
      } as const,
      default: {
        elevation: 16,
        shadowColor: "#0f172a",
        shadowOpacity: 0.12,
        shadowOffset: { width: 0, height: -4 },
        shadowRadius: 24,
      },
    }),
  },
  sheetTitle: {
    color: C.text,
    marginBottom: 6,
    fontWeight: "700",
  },
  sheetRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 13,
    paddingHorizontal: 12,
    borderRadius: 14,
    borderWidth: StyleSheet.hairlineWidth,
  },
  sheetRowIdle: {
    backgroundColor: "transparent",
    borderColor: "rgba(228, 231, 236, 0.85)",
  },
  sheetRowActive: {
    backgroundColor: "rgba(10, 143, 122, 0.08)",
    borderColor: C.brand,
  },
  sheetRowPressed: {
    opacity: 0.92,
  },
  sheetRowLabel: {
    fontWeight: "600",
  },
});
