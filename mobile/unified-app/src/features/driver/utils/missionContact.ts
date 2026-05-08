import { Alert, Linking, Platform } from "react-native";
import type { DriverMission } from "../types";

/**
 * Téléphone E.164 simplifié : alignement strict sur operations-app
 * (`utils/phone.ts`). Backend impose `^\+?\d{7,15}$`.
 */
export const PHONE_REGEX = /^\+?\d{7,15}$/;

/**
 * Normalise une chaîne téléphone (mêmes règles qu'operations-app).
 * - trim, supprime espaces/tirets/parenthèses/points
 * - vide => null
 * - 00 au début => +
 * - ne garde que chiffres + '+' en première position
 * - +41 (0)79… => +4179…
 */
export function normalizePhone(raw: string | null | undefined): string | null {
  if (raw == null) return null;
  let s = String(raw).trim();
  if (s === "") return null;
  if (s.startsWith("00")) s = "+" + s.slice(2);
  const hasPlus = s.startsWith("+");
  let digits = s.replace(/\D/g, "");
  if (digits.length === 0) return null;
  if (hasPlus && digits.length > 3 && digits[2] === "0") {
    digits = digits.slice(0, 2) + digits.slice(3);
  }
  return hasPlus ? "+" + digits : digits;
}

/** Valide une chaîne normalisée (format backend). */
export function isValidPhone(normalized: string | null | undefined): boolean {
  if (normalized == null || normalized === "") return false;
  return PHONE_REGEX.test(normalized);
}

/**
 * Premier numéro joignable selon la priorité d'operations-app
 * (`client.contact_phone` > `client.phone` > `client.gp_phone`
 *  > racine `contact_phone` > `phone` > `gp_phone` > `client_phone` legacy).
 */
export function getCallablePhoneFromMission(mission: DriverMission): string | null {
  if (mission == null) return null;
  const nested = (mission.client as Record<string, unknown> | undefined) ?? null;
  const candidates: (string | null | undefined)[] = [
    nested?.contact_phone as string | null | undefined,
    nested?.phone as string | null | undefined,
    nested?.gp_phone as string | null | undefined,
    mission.contact_phone as string | null | undefined,
    mission.phone as string | null | undefined,
    mission.gp_phone as string | null | undefined,
    mission.client_phone as string | null | undefined,
  ];
  for (const raw of candidates) {
    const normalized = normalizePhone(raw ?? null);
    if (normalized && isValidPhone(normalized)) return normalized;
  }
  return null;
}

/** Construit le lien `tel:` (numéro déjà normalisé). */
export function buildCallLink(phone: string): string {
  return `tel:${phone.replace(/\s+/g, "")}`;
}

/**
 * URL maps native par plateforme (operations-app `services/deepLinks.ts`) :
 * - Android : `geo:0,0?q=` ouvre le gestionnaire maps par défaut (Google Maps, Waze…).
 * - iOS    : `maps://?daddr=` ouvre Plans avec directions.
 */
function buildNativeNavigationUrl(destination: string): string {
  const encoded = encodeURIComponent(destination);
  if (Platform.OS === "android") return `geo:0,0?q=${encoded}`;
  return `maps://?daddr=${encoded}`;
}

/** Repli web qui marche dans n'importe quel navigateur. */
export function buildMapsDirectionsUrl(destination: string): string | null {
  const q = destination.trim();
  if (!q) return null;
  return `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(q)}`;
}

/** Ouverture URL avec pré-check `canOpenURL` (iOS) et `tel:` direct. */
async function safeOpenURL(url: string): Promise<boolean> {
  try {
    if (Platform.OS === "ios" && !url.startsWith("tel:")) {
      const canOpen = await Linking.canOpenURL(url);
      if (!canOpen) return false;
    }
    await Linking.openURL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Ouvre l'app téléphone : `Linking.openURL` (natif), `window.open` + alerte sur web
 * (aligné comportement operations-app `(tabs)/mission.tsx` `onCall`).
 */
export async function safeCall(rawPhone: string): Promise<boolean> {
  const normalized = normalizePhone(rawPhone);
  if (!normalized || !isValidPhone(normalized)) return false;
  const url = buildCallLink(normalized);
  if (Platform.OS === "web") {
    try {
      const w = (globalThis as { open?: (u: string) => void }).open;
      if (typeof w === "function") {
        w(url);
      } else {
        await Linking.openURL(url);
      }
      Alert.alert(
        "Appel",
        "Ouverture de l'appel… Si rien ne se passe, aucun logiciel d'appel n'est peut-être configuré sur cet appareil."
      );
      return true;
    } catch {
      return false;
    }
  }
  const opened = await safeOpenURL(url);
  if (!opened) {
    Alert.alert("Appel impossible", "Impossible d'ouvrir l'application téléphone.");
  }
  return opened;
}

/**
 * Ouvre la navigation : tente l'app native d'abord (Android `geo:`, iOS `maps://`),
 * puis bascule sur Google Maps web si l'app native n'est pas disponible.
 */
export async function openNavigation(destination: string): Promise<boolean> {
  const dest = destination.trim();
  if (!dest) return false;

  if (Platform.OS !== "web") {
    const nativeUrl = buildNativeNavigationUrl(dest);
    const opened = await safeOpenURL(nativeUrl);
    if (opened) return true;
  }
  const webUrl = buildMapsDirectionsUrl(dest);
  if (!webUrl) return false;
  return safeOpenURL(webUrl);
}
