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
import { useRouter, type Href } from "expo-router";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { shouldShowCompanyDriverContextSwitch } from "../../../core/contextSwitchPolicy";
import { useSession } from "../../../core/sessionProvider";
import { useCompanyMessageHubUnreadBadge } from "../messages/hooks";
import { RadialActionMenu, type RadialAction } from "../../../components/RadialActionMenu";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

const C = {
  border: "rgba(148, 163, 184, 0.22)",
  text: "#0F172A",
  textMuted: "#64748B",
  brand: "#00796B",
  brandDark: "#00796B",
} as const;

/** Web / mobile web : pas de contour rectangulaire au focus, tap ou « hover » sur Pressable. */
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

const CONTEXT_SWITCH_TIMEOUT_MS = 45_000;

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

const HIDDEN_SHEET_ROUTES: {
  name: "clients-facturation" | "settings";
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  href: Href;
}[] = [
  {
    name: "clients-facturation",
    label: "Clients & facturation",
    icon: "reader-outline",
    href: "/(app)/(company)/clients-facturation" as Href,
  },
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
 * (patron 2 + bouton rond + 2). « Autres » ouvre Clients & facturation et Paramètres.
 *
 * Onglets : icônes + badge (chat) ; pas de libellé sous la barre — `label` sert à l’accessibilité.
 */
export function CompanyFloatingTabBar({ state, navigation }: BottomTabBarProps) {
  const { usableWidth, bottomInset, horizontalPadding } = useAppViewport();
  const { isLargeText } = useAccessibilityScale();
  const router = useRouter();
  const { activeContext, bootstrap, changeContext } = useSession();
  const canCreateRide = useCanCreateCompanyRide();
  const [switchPending, setSwitchPending] = useState(false);
  const [switchMessage, setSwitchMessage] = useState<string | null>(null);
  const chatUnread = useCompanyMessageHubUnreadBadge();
  const current = useActiveRouteName(state);
  const maxBarWidth = Math.min(480, usableWidth - 2 * horizontalPadding);
  const focusedFromSheet = HIDDEN_SHEET_ROUTES.some((x) => x.name === current);
  const bottomPad = computeCompanyFloatingBottomPad(bottomInset);
  const totalBarAreaHeight = 64 + bottomInset;
  const primaryDriverContext = useMemo(() => {
    const contexts = bootstrap?.available_contexts ?? [];
    return (
      contexts.find(
        (ctx) => ctx.context_type === "driver" && ctx.allow_mobile_context_switch === true
      ) ?? null
    );
  }, [bootstrap?.available_contexts]);

  const canSwitchToDriver = useMemo(
    () =>
      activeContext?.context_type === "company" &&
      primaryDriverContext != null &&
      shouldShowCompanyDriverContextSwitch(
        activeContext,
        primaryDriverContext,
        bootstrap?.user?.role
      ),
    [activeContext, bootstrap?.user?.role, primaryDriverContext]
  );

  const radialActions = useMemo<RadialAction[]>(() => {
    const items: RadialAction[] = [
      {
        key: "clients-facturation",
        label: "Clients & facturation",
        icon: <Ionicons name="reader-outline" size={20} color="#FFFFFF" />,
        color: "#00796B",
        onPress: () => {
          void router.push("/(app)/(company)/clients-facturation" as Href);
        },
      },
      {
        key: "settings",
        label: "Paramètres",
        icon: <Ionicons name="settings-outline" size={20} color="#FFFFFF" />,
        color: "#0E7490",
        onPress: () => {
          void router.push("/(app)/(company)/settings" as Href);
        },
      },
    ];
    if (canSwitchToDriver && primaryDriverContext) {
      items.push({
        key: "driver-context",
        label: switchPending ? "Bascule..." : "Contexte chauffeur",
        icon: <Ionicons name="swap-horizontal-outline" size={20} color="#FFFFFF" />,
        color: "#1D4D8F",
        disabled: switchPending,
        onPress: () => {
          void (async () => {
            if (switchPending) return;
            setSwitchPending(true);
            setSwitchMessage("Bascule vers l'espace chauffeur…");
            const switchWork = withTimeout(
              changeContext(primaryDriverContext.context_id),
              CONTEXT_SWITCH_TIMEOUT_MS,
              "La bascule a pris trop de temps. Vérifiez votre connexion et réessayez."
            );
            router.replace("/(app)/(driver)" as Href);
            try {
              await switchWork;
              setSwitchMessage(null);
            } catch (error) {
              setSwitchMessage(
                error instanceof Error
                  ? error.message
                  : "Impossible de basculer vers le contexte chauffeur."
              );
            } finally {
              setSwitchPending(false);
            }
          })();
        },
      });
    }
    return items;
  }, [canSwitchToDriver, changeContext, primaryDriverContext, router, switchPending]);

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
            label="Dashboard"
            icon="speedometer-outline"
            active={current === "dashboard" && !focusedFromSheet}
            onPress={() => {
              navigation.navigate("dashboard" as never);
            }}
          />
          <BarTabButton
            label="Courses"
            icon="car-outline"
            active={current === "rides" && !focusedFromSheet}
            onPress={() => {
              navigation.navigate("rides" as never);
            }}
          />
          <View style={styles.barSlot}>
            <Pressable
              onPress={() => {
                if (canCreateRide) {
                  void router.push({ pathname: "/(app)/(company)/rides", params: { create: "1" } } as Href);
                }
              }}
              disabled={!canCreateRide}
              accessibilityLabel="Nouvelle course"
              accessibilityRole="button"
              android_ripple={
                Platform.OS === "android"
                  ? { color: "rgba(255, 255, 255, 0.35)", borderless: true }
                  : undefined
              }
              style={({ pressed }) => [
                styles.fabOuter,
                !canCreateRide && styles.fabOuterDisabled,
                pressed && canCreateRide && styles.fabOuterPressed,
                Platform.OS === "web" ? PRESSABLE_WEB_SUPPRESS_SQUARE_HALO : null,
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
              navigation.navigate("messages" as never);
            }}
          />
          <View style={styles.barSlot}>
            <RadialActionMenu
              inline
              actions={radialActions}
              triggerVariant="tab"
              actionsLayout="vertical"
              mainIcon={<Ionicons name="grid-outline" size={22} color={focusedFromSheet ? C.brand : C.textMuted} />}
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
        <View style={styles.switchMessageFloating} accessibilityLiveRegion="polite">
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
  /** Cinq colonnes égales (Dashboard, Courses, FAB, Chat, menu). */
  barSlot: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 48,
  },
  tabHitPressed: {
    opacity: Platform.OS === "ios" ? 0.88 : 1,
  },
  /** Hit target 44×44, forme pilule — ripple / focus limités à cette forme. */
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
  badgeText: {
    fontWeight: "800",
    color: C.brand,
    fontSize: FONT_SIZE.px10,
    lineHeight: 12,
  },
  fabOuter: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: C.brandDark,
    alignItems: "center",
    justifyContent: "center",
    alignSelf: "center",
    ...Platform.select({
      web: {
        boxShadow: "0 6px 18px rgba(0, 121, 107, 0.35)",
      } as const,
      default: {
        elevation: 8,
        shadowColor: "#00796B",
        shadowOpacity: 0.35,
        shadowOffset: { width: 0, height: 4 },
        shadowRadius: 10,
      },
    }),
  },
  fabOuterDisabled: {
    backgroundColor: "rgba(0, 121, 107, 0.38)",
    ...Platform.select({
      web: { boxShadow: "none" } as const,
      default: { elevation: 0, shadowOpacity: 0 },
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
    backgroundColor: "rgba(0, 121, 107, 0.08)",
    borderColor: C.brand,
  },
  sheetRowPressed: {
    opacity: 0.92,
  },
  sheetRowLabel: {
    fontWeight: "600",
  },
  contextSwitchBtn: {
    minHeight: 36,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 1,
    backgroundColor: C.brand,
    borderColor: C.brand,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 8,
  },
  contextSwitchBtnDisabled: {
    opacity: 0.45,
  },
  contextSwitchBtnPressed: {
    opacity: 0.9,
  },
  contextSwitchBtnText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
  },
  contextSwitchMessage: {
    color: "#B42318",
    lineHeight: 16,
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
});
