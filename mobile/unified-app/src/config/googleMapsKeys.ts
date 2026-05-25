/**
 * Résolution des clés **client** Maps (web / natif Expo).
 * La clé serveur `GOOGLE_MAPS_API_KEY` (backend/.env uniquement) ne doit jamais
 * apparaître ici ni sous EXPO_PUBLIC_* / REACT_APP_*.
 */
import { Platform } from "react-native";
import { parseGoogleMapsLibraryList } from "../shared/google-maps/bootstrap";

/** Librairies JS réellement utilisées par la carte flotte (évite `places` si non activé GCP). */
const FLEET_WEB_MAP_LIBRARIES = new Set(["maps", "marker", "routes"]);

function trimEnv(name: string): string {
  const v =
    name === "EXPO_PUBLIC_GOOGLE_MAPS_API_KEY"
      ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY
      : name === "EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY"
        ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY
        : name === "EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY"
          ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY
          : name === "EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES"
            ? process.env.EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES
            : undefined;
  return typeof v === "string" ? v.trim() : "";
}

/** Filtre les placeholders / gabarits `.env.example`. */
export function isPlausibleGoogleMapsBrowserKey(k: string): boolean {
  const t = k.trim();
  if (t.length < 20) return false;
  const lower = t.toLowerCase();
  if (lower.includes("ta_clef")) return false;
  if (lower.includes("google_maps_js")) return false;
  if (lower.includes("your_api")) return false;
  if (lower.includes("replace_me")) return false;
  if (lower.includes("changeme")) return false;
  if (lower.includes("example_key")) return false;
  if (lower.includes("placeholder")) return false;
  return true;
}

export type GoogleMapsWebKeyIssue =
  | "ok"
  | "missing"
  | "invalid"
  | "android_only_configured";

/** Diagnostic clé web (Messages UI carte navigateur). */
export function diagnoseGoogleMapsWebKeyIssue(): GoogleMapsWebKeyIssue {
  const raw = trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
  if (!raw) {
    const hasNativeOnly =
      trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY").length > 0 ||
      trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY").length > 0;
    return hasNativeOnly ? "android_only_configured" : "missing";
  }
  if (!isPlausibleGoogleMapsBrowserKey(raw)) return "invalid";
  return "ok";
}

/** Message d’aide selon `diagnoseGoogleMapsWebKeyIssue()`. */
export function formatGoogleMapsWebKeyHelpMessage(issue: GoogleMapsWebKeyIssue): string {
  switch (issue) {
    case "ok":
      return "";
    case "android_only_configured":
      return (
        "Clé Android/iOS détectée, mais pas de clé web. Définissez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY " +
        "(Maps JavaScript API, restrictions HTTP referrer pour l’URL du bundle web), puis reconstruisez le bundle."
      );
    case "invalid":
      return "Clé Google Maps web invalide ou placeholder — configurez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.";
    case "missing":
    default:
      return "Définissez EXPO_PUBLIC_GOOGLE_MAPS_API_KEY pour la carte web (Maps JavaScript API).";
  }
}

/** Expo Web : clé **Maps JavaScript API** (restrictions HTTP referrer). */
export function resolveGoogleMapsWebApiKey(): string | undefined {
  const issue = diagnoseGoogleMapsWebKeyIssue();
  if (issue !== "ok") return undefined;
  const k = trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
  return k.length > 0 ? k : undefined;
}

/**
 * Librairies chargées pour la carte flotte web.
 * Filtre `EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES` (ex. retire `places` si l’API n’est pas activée).
 */
export function resolveFleetMapsLibraryList(): string[] {
  const fleetOnly = trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_FLEET_LIBRARIES");
  const raw = fleetOnly || trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_LIBRARIES") || "marker";
  const parsed = parseGoogleMapsLibraryList(raw);
  const filtered = parsed.filter((name) => FLEET_WEB_MAP_LIBRARIES.has(name));
  return filtered.length > 0 ? filtered : ["marker"];
}

/**
 * Android / iOS natif : clé **Maps SDK + Directions** (restrictions appli).
 * Préférer `EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY` / `EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY`.
 * Sinon repli sur `EXPO_PUBLIC_GOOGLE_MAPS_API_KEY` (config historique une seule clé).
 */
export function resolveGoogleMapsNativeApiKey(): string | undefined {
  if (Platform.OS === "android") {
    const android = trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_ANDROID_API_KEY");
    if (android.length > 0) return android;
  }
  if (Platform.OS === "ios") {
    const ios = trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_IOS_API_KEY");
    if (ios.length > 0) return ios;
  }
  const legacy = trimEnv("EXPO_PUBLIC_GOOGLE_MAPS_API_KEY");
  return legacy.length > 0 ? legacy : undefined;
}
