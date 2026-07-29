import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { Href, useRouter, useSegments } from "expo-router";
import { useCallback, useMemo } from "react";
import { Platform, Pressable, View } from "react-native";
import { AppText } from "../../../design/ui/AppText";
import {
  BaseFloatingBar,
  brandPrimary,
  computeClientFloatingBottomPad,
  FLOATING_BAR_FALLBACK_INNER,
  useFloatingBarClearance,
} from "../../../design/responsive";
import { useAccessibilityScale } from "../../../design/responsive/useAccessibilityScale";
import { useAppViewport } from "../../../design/responsive/useAppViewport";

const BRAND = brandPrimary;
const BRAND_STRONG = "#067A66";

const C = {
  iconMuted: "#64748b",
  labelMuted: "#64748b",
} as const;

const HREF = {
  home: "/(app)/(client)" as Href,
  bookings: "/(app)/(client)/bookings" as Href,
  newBooking: "/(app)/(client)/booking/new" as Href,
  account: "/(app)/(client)/account" as Href,
} as const;

/**
 * @deprecated Fallback hors provider — préférer {@link useClientBottomContentPadding}.
 */
export const CLIENT_FLOATING_BAR_BASE_HEIGHT = FLOATING_BAR_FALLBACK_INNER.client;

export function useClientFloatingBarVisible(): boolean {
  const segments = useSegments();
  return useMemo(() => !shouldHideClientFloatingBar(segments), [segments]);
}

export function useClientBottomContentPadding(): number {
  const { bottomInset } = useAppViewport();
  const segments = useSegments();
  const visible = !shouldHideClientFloatingBar(segments);
  const bottomPad = computeClientFloatingBottomPad(bottomInset);
  const barClearance = useFloatingBarClearance("client", bottomPad);
  if (!visible) return Math.max(24, bottomInset + 8);
  return barClearance;
}

/** Masquer la barre sur les écrans « plein flux » (paiement, fiche course). La création garde le menu. */
function shouldHideClientFloatingBar(segments: string[]): boolean {
  if (segments.includes("payment")) return true;
  const i = segments.indexOf("booking");
  if (i === -1) return false;
  const next = segments[i + 1];
  if (next && /^\d+$/.test(next)) return true;
  return false;
}

function useActiveFlags() {
  const segments = useSegments() as readonly string[];
  return useMemo(() => {
    const clientIdx = segments.lastIndexOf("(client)");
    const rest = clientIdx >= 0 ? segments.slice(clientIdx + 1) : [];
    const top = rest[0];
    const onNewBooking = top === "booking" && rest[1] === "new";
    const onAccount = top === "account";
    const onBookings = top === "bookings";
    const onHome =
      (!top || top === "index") && !onBookings && !onAccount && !onNewBooking;
    return {
      home: onHome,
      wallet: onBookings,
      settings: onAccount,
      profile: onAccount,
      newBooking: onNewBooking,
    };
  }, [segments]);
}

async function hapticLight() {
  if (Platform.OS === "ios") {
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    } catch {
      /* ignore */
    }
  }
}

/**
 * Barre inférieure flottante (pilule) type Flowbite « application bar » :
 * Accueil, Portefeuille → réservations, CTA central Réserver, Réglages, Profil.
 *
 * Texte (UI critique) : libellés courts avec `maxFontSizeMultiplier` bas, `numberOfLines` + ellipsize ;
 * pas de `allowFontScaling={false}`. Grande police : minHeight barre via `isLargeText`.
 */
export function ClientFloatingAppBar() {
  const { usableWidth, bottomInset, horizontalPadding } = useAppViewport();
  const { isLargeText } = useAccessibilityScale();
  const router = useRouter();
  const active = useActiveFlags();
  const maxBarWidth = Math.min(512, usableWidth - 2 * horizontalPadding);
  const bottomPad = computeClientFloatingBottomPad(bottomInset);

  const goTab = useCallback(
    (href: Href) => {
      void hapticLight();
      router.replace(href);
    },
    [router]
  );

  const goNewBooking = useCallback(() => {
    void hapticLight();
    if (active.newBooking) {
      router.replace(HREF.newBooking);
      return;
    }
    router.push(HREF.newBooking);
  }, [router, active.newBooking]);

  return (
    <BaseFloatingBar
      paddingBottom={bottomPad}
      maxBarWidth={maxBarWidth}
      horizontalPadding={horizontalPadding}
      preset="client"
      minInnerHeight={64}
      minInnerHeightLargeText={72}
      isLargeText={isLargeText}
    >
      <BarIconTab
        label="Accueil"
        icon="home-outline"
        active={active.home}
        onPress={() => goTab(HREF.home)}
        roundedLeft
      />
      <BarIconTab
        label="Courses"
        accessibilityLabel="Mes réservations"
        icon="wallet-outline"
        active={active.wallet}
        onPress={() => goTab(HREF.bookings)}
      />
      <View style={{ width: 56, alignItems: "center", justifyContent: "center" }}>
        <Pressable
          onPress={goNewBooking}
          accessibilityLabel="Réserver un transport"
          accessibilityState={{ selected: active.newBooking }}
          style={({ pressed }) => [
            {
              width: 48,
              height: 48,
              borderRadius: 24,
              backgroundColor: BRAND,
              alignItems: "center",
              justifyContent: "center",
              borderWidth: active.newBooking ? 3 : 0,
              borderColor: "rgba(255, 255, 255, 0.95)",
            },
            Platform.select({
              web: { boxShadow: "0 2px 8px rgba(10, 126, 164, 0.35)" } as const,
              default: {
                elevation: 4,
                shadowColor: BRAND,
                shadowOpacity: 0.35,
                shadowOffset: { width: 0, height: 2 },
                shadowRadius: 6,
              },
            }),
            pressed && { backgroundColor: BRAND_STRONG },
          ]}
        >
          <Ionicons name="add" size={28} color="#FFFFFF" />
        </Pressable>
      </View>
      <BarIconTab
        label="Réglages"
        icon="options-outline"
        active={active.settings}
        onPress={() => goTab(HREF.account)}
      />
      <BarIconTab
        label="Profil"
        icon="person-outline"
        active={active.profile}
        onPress={() => goTab(HREF.account)}
        roundedRight
      />
    </BaseFloatingBar>
  );
}

function BarIconTab({
  label,
  accessibilityLabel,
  icon,
  active,
  onPress,
  roundedLeft,
  roundedRight,
}: {
  label: string;
  accessibilityLabel?: string;
  icon: keyof typeof Ionicons.glyphMap;
  active: boolean;
  onPress: () => void;
  roundedLeft?: boolean;
  roundedRight?: boolean;
}) {
  const radius = 999;
  return (
    <Pressable
      onPress={onPress}
      accessibilityLabel={accessibilityLabel ?? label}
      style={({ pressed }) => [
        {
          flex: 1,
          alignItems: "center",
          justifyContent: "center",
          paddingVertical: 6,
          paddingHorizontal: 4,
          borderTopLeftRadius: roundedLeft ? radius : 8,
          borderBottomLeftRadius: roundedLeft ? radius : 8,
          borderTopRightRadius: roundedRight ? radius : 8,
          borderBottomRightRadius: roundedRight ? radius : 8,
        },
        pressed && { backgroundColor: "rgba(15, 23, 42, 0.06)" },
      ]}
    >
      <Ionicons name={icon} size={24} color={active ? BRAND : C.iconMuted} />
      <AppText
        variant="caption"
        numberOfLines={1}
        maxFontSizeMultiplier={1.28}
        ellipsizeMode="tail"
        style={{
          marginTop: 2,
          fontWeight: "700",
          color: active ? BRAND : C.labelMuted,
        }}
      >
        {label}
      </AppText>
    </Pressable>
  );
}
