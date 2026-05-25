import { useMemo, useState } from "react";
import { Platform, Pressable, StyleSheet, View } from "react-native";
import {
  BaseFloatingBar,
  computeCompanyFloatingBottomPad,
} from "../../../design/responsive";
import { AppText } from "../../../design/ui/AppText";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { useAppViewport } from "../../../design/responsive/useAppViewport";
import { Ionicons } from "@expo/vector-icons";
import type { BottomTabBarProps } from "@react-navigation/bottom-tabs";
import { useRouter, useSegments } from "expo-router";
import { shouldShowCompanyDriverContextSwitch } from "../../../core/contextSwitchPolicy";
import { useSession } from "../../../core/sessionProvider";
import { useDriverMessageHubUnreadBadge } from "../messages/hooks";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { useDriverChatUnreadCount } from "../chatHooks";
import { RadialActionMenu, type RadialAction } from "../../../components/RadialActionMenu";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const C = {
  textMuted: "#7A808A",
  brand: "#00796B",
  messagesActive: "#0A8F7A",
} as const;

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

/** Padding scroll pour que le dernier contenu ne passe pas sous la pilule (company dashboard ≈ 80). */
export const DRIVER_FLOATING_TAB_SCROLL_PADDING = 96;

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
  messages: boolean;
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
    const messages =
      last === "messages" ||
      prev === "messages" ||
      (last === "chat" && prev === "(driver)");
    const moreRoute =
      last === "schedule" ? "schedule" : last === "profile" ? "profile" : null;

    return { home, trips, messages, moreRoute };
  }, [segments]);
}

/**
 * Barre d’onglets flottante alignée sur {@link CompanyFloatingTabBar} :
 * Accueil, Courses, FAB (liste missions), Chat + menu radial « Autres ».
 */
export function DriverFloatingTabBar({ navigation }: BottomTabBarProps) {
  const { usableWidth, bottomInset, horizontalPadding } = useAppViewport();
  const { isLargeText } = useAccessibilityScale();
  const router = useRouter();
  const { activeContext, bootstrap, changeContext } = useSession();
  const [switchPending, setSwitchPending] = useState(false);
  const [switchMessage, setSwitchMessage] = useState<string | null>(null);
  const hubUnread = useDriverMessageHubUnreadBadge();
  const legacyChatUnread = useDriverChatUnreadCount();
  const messagesUnread = isFeatureEnabled("driver_messages_hub_enabled")
    ? hubUnread
    : legacyChatUnread;
  const tab = useDriverTabHighlight();
  const maxBarWidth = Math.min(480, usableWidth - 2 * horizontalPadding);
  const bottomPad = computeCompanyFloatingBottomPad(bottomInset);
  const totalBarAreaHeight = 64 + bottomInset;
  const focusedFromSheet = tab.moreRoute != null;
  const targetCompanyContext = useMemo(() => {
    const contexts = bootstrap?.available_contexts ?? [];
    return (
      contexts.find(
        (ctx) =>
          ctx.context_type === "company" && ctx.allow_mobile_context_switch === true
      ) ?? null
    );
  }, [bootstrap?.available_contexts]);

  const canSwitchToCompany = useMemo(
    () =>
      activeContext?.context_type === "driver" &&
      targetCompanyContext != null &&
      shouldShowCompanyDriverContextSwitch(
        activeContext,
        targetCompanyContext,
        bootstrap?.user?.role
      ),
    [activeContext, bootstrap?.user?.role, targetCompanyContext]
  );

  const radialActions = useMemo<RadialAction[]>(() => {
    const items: RadialAction[] = [
      {
        key: "schedule",
        label: "Planning",
        icon: <Ionicons name="calendar-outline" size={20} color="#FFFFFF" />,
        color: "#00796B",
        onPress: () => {
          void router.push("/(app)/(driver)/schedule");
        },
      },
      {
        key: "profile",
        label: "Profil chauffeur",
        icon: <Ionicons name="person-outline" size={20} color="#FFFFFF" />,
        color: "#0E7490",
        onPress: () => {
          void router.push("/(app)/(driver)/profile");
        },
      },
    ];
    if (canSwitchToCompany && targetCompanyContext) {
      items.push({
        key: "company-context",
        label: switchPending ? "Bascule..." : "Espace entreprise",
        icon: <Ionicons name="swap-horizontal-outline" size={20} color="#FFFFFF" />,
        color: "#1D4D8F",
        disabled: switchPending,
        onPress: () => {
          void (async () => {
            if (switchPending) return;
            setSwitchPending(true);
            setSwitchMessage("Bascule vers l'espace entreprise…");
            const switchWork = withTimeout(
              changeContext(targetCompanyContext.context_id),
              45_000,
              "La bascule a pris trop de temps. Vérifiez votre connexion et réessayez."
            );
            router.replace("/(app)/(company)/dashboard");
            try {
              await switchWork;
              setSwitchMessage(null);
            } catch (error) {
              setSwitchMessage(
                error instanceof Error
                  ? error.message
                  : "Impossible de revenir à l’espace entreprise."
              );
            } finally {
              setSwitchPending(false);
            }
          })();
        },
      });
    }
    return items;
  }, [canSwitchToCompany, changeContext, router, switchPending, targetCompanyContext]);

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
            navigation.navigate("index" as never);
          }}
        />
        <BarTabButton
          label="Courses"
          icon="car-outline"
          active={tab.trips && !focusedFromSheet}
          onPress={() => {
            navigation.navigate("trips" as never);
          }}
        />
        <View style={styles.barSlot}>
          <Pressable
            onPress={() => {
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
          label="Messages"
          icon={
            tab.messages && !focusedFromSheet
              ? "chatbubble-ellipses"
              : "chatbubble-ellipses-outline"
          }
          active={tab.messages && !focusedFromSheet}
          badgeCount={messagesUnread}
          activeColor={C.messagesActive}
          onPress={() => {
            navigation.navigate("messages" as never);
          }}
        />
        <View style={styles.barSlot}>
          <RadialActionMenu
            inline
            actions={radialActions}
            triggerVariant="tab"
            actionsLayout="vertical"
            mainIcon={<Ionicons name="grid-outline" size={22} color={C.textMuted} />}
            openIcon={<Ionicons name="close-outline" size={22} color={C.brand} />}
            position="bottomRight"
            radius={68}
            verticalSpacing={40}
            verticalExtraSpacing={14}
            actionsOffsetX={0}
            actionsOffsetY={-20}
            showLabels={false}
            accessibilityLabel="Autres écrans"
          />
        </View>
      </BaseFloatingBar>
      {switchMessage ? (
        <View pointerEvents="none" style={styles.switchMessageFloating}>
          <AppText variant="caption" style={styles.contextSwitchMessage}>
            {switchMessage}
          </AppText>
        </View>
      ) : null}
    </>
  );
}

function BarTabButton({
  label,
  icon,
  active,
  onPress,
  badgeCount = 0,
  activeColor = C.brand,
}: {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  active: boolean;
  onPress: () => void;
  badgeCount?: number;
  activeColor?: string;
}) {
  const a11y =
    badgeCount > 0
      ? `${label}, ${badgeCount} non lu${badgeCount > 1 ? "s" : ""}`
      : label;
  return (
    <View style={styles.barSlot}>
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
          <Ionicons name={icon} size={22} color={active ? activeColor : C.textMuted} />
          {badgeCount > 0 ? (
            <View
              style={[styles.badge, active && styles.badgeActive]}
              accessibilityLabel={`${badgeCount}`}
              accessibilityRole="text"
              accessible
            >
              <AppText
                variant="caption"
                numberOfLines={1}
                maxFontSizeMultiplier={1.25}
                style={[styles.badgeText, active && styles.badgeTextActive]}
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
  /** Cinq colonnes égales (Accueil, Courses, FAB, Messages, menu). */
  barSlot: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 48,
  },
  tabHitPressed: {
    opacity: Platform.OS === "ios" ? 0.88 : 1,
  },
  tabPressable: {
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    width: 44,
    height: 44,
    overflow: "hidden",
  },
  tabIconShellActive: {
    backgroundColor: "rgba(0, 121, 107, 0.14)",
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
    backgroundColor: "rgba(0, 121, 107, 0.2)",
    borderWidth: 1,
    borderColor: C.brand,
    alignItems: "center",
    justifyContent: "center",
  },
  badgeActive: {
    backgroundColor: C.messagesActive,
    borderColor: C.messagesActive,
  },
  badgeText: {
    fontWeight: "800",
    color: C.brand,
    fontSize: FONT_SIZE.px10,
    lineHeight: 12,
  },
  badgeTextActive: {
    color: "#fff",
  },
  fabOuter: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.brand,
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "center",
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
  switchMessageFloating: {
    position: "absolute",
    right: 16,
    bottom: 140,
    maxWidth: 220,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: "rgba(180, 35, 24, 0.36)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 10,
    paddingVertical: 8,
    ...Platform.select({
      web: { boxShadow: "0 2px 8px rgba(15, 23, 42, 0.14)" } as const,
      default: {
        elevation: 2,
        shadowColor: "#0f172a",
        shadowOpacity: 0.1,
        shadowOffset: { width: 0, height: 1 },
        shadowRadius: 4,
      },
    }),
  },
  contextSwitchMessage: {
    color: "#B42318",
    lineHeight: 16,
  },
});
