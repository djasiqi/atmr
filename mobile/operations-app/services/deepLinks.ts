import * as Linking from "expo-linking";
import { Platform, Alert } from "react-native";
import { getLogger } from "@/utils/logger";

const log = getLogger("DeepLinks");
const SCHEME = "atmr";

export function buildQuickActionLink(params: {
  fromNavigation?: boolean;
  bookingId?: string;
}): string {
  const qs = new URLSearchParams();
  if (params.fromNavigation) qs.set("fromNavigation", "true");
  if (params.bookingId) qs.set("bookingId", params.bookingId);
  const query = qs.toString();
  return `${SCHEME}://mission/quick-action${query ? `?${query}` : ""}`;
}

export function buildCallLink(phone: string): string {
  const cleaned = phone.replace(/\s+/g, "");
  return `tel:${cleaned}`;
}

/**
 * Build a platform-native navigation URL.
 * - Android: `geo:0,0?q=` → opens user's default maps app (Google Maps, Waze, etc.)
 * - iOS: `maps://?daddr=` → opens Apple Maps with directions
 */
function buildNativeNavigationUrl(destination: string): string {
  const encoded = encodeURIComponent(destination);
  if (Platform.OS === "android") {
    return `geo:0,0?q=${encoded}`;
  }
  return `maps://?daddr=${encoded}`;
}

/**
 * Web fallback URL that works in any browser.
 */
function buildWebNavigationUrl(destination: string): string {
  return `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}`;
}

/**
 * Open a URL safely with canOpenURL pre-check.
 * Returns true if opened, false if not supported.
 */
export async function safeOpenURL(url: string): Promise<boolean> {
  try {
    if (Platform.OS === "ios" && !url.startsWith("tel:")) {
      const canOpen = await Linking.canOpenURL(url);
      if (!canOpen) {
        log.warn("cannot open url", { url });
        return false;
      }
    }
    await Linking.openURL(url);
    return true;
  } catch (error) {
    log.error("failed to open url", { url, error });
    return false;
  }
}

export async function safeCall(phone: string): Promise<boolean> {
  const url = buildCallLink(phone);
  const opened = await safeOpenURL(url);
  if (!opened) {
    Alert.alert("Appel impossible", "Impossible d'ouvrir l'application téléphone.");
  }
  return opened;
}

/**
 * Open the user's default navigation/maps app with directions to the destination.
 * - Android: opens default maps handler (Google Maps, Waze, Here, etc.)
 * - iOS: opens Apple Maps
 * - Fallback: opens Google Maps in browser
 */
export async function openNavigation(destination: string): Promise<boolean> {
  if (!destination) return false;

  const nativeUrl = buildNativeNavigationUrl(destination);
  const opened = await safeOpenURL(nativeUrl);
  if (opened) return true;

  const webUrl = buildWebNavigationUrl(destination);
  return safeOpenURL(webUrl);
}

/** @deprecated Use `openNavigation` instead */
export const openGoogleMaps = openNavigation;
