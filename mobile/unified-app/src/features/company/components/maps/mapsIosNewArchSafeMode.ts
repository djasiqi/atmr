import { Platform } from "react-native";

/**
 * Hotfix iOS — désactive les <Marker> de react-native-maps qui rendent des
 * enfants React custom (View, Text, Animated.View, etc.) sur iOS.
 *
 * Cause : sous New Architecture, l'interop legacy ViewManager
 * (`RCTLegacyViewManagerInteropComponentView.finalizeUpdates`) peut tenter
 * d'insérer une sous-vue `nil` dans `__NSArrayM insertObject:atIndex:`,
 * ce qui lève une NSInvalidArgumentException fatale (Sentry
 * 32b6e2e9e86243d6a1d80d8905165368, 55e711c3… build 1.0.5+61, iPhone 16 Pro Max, iOS 26.5).
 *
 * Le crash arrive pendant un re-render de marker custom (pulse, badge) sous
 * activité chauffeur (switch context, mises à jour GPS/statut rapprochées).
 *
 * Mitigation OTA conservatrice : sur iOS, on ne monte plus que des markers
 * raster `icon`/`image` (sans children React). La pulsation et le badge
 * compteur cluster sont temporairement masqués sur iOS uniquement, en
 * attendant un fix natif (build) ou un upgrade compatible New Arch.
 *
 * Android reste inchangé (le crash est spécifique à l'interop iOS).
 */
export const IOS_MAP_NO_CUSTOM_MARKER_CHILDREN = Platform.OS === "ios";

/**
 * Garde-fou coordonnées : ne JAMAIS monter un `<Marker>` avec lat/lng
 * `NaN`, `Infinity`, `null` ou `undefined`. Une coordonnée invalide peut
 * provoquer le même type de crash natif côté iOS / Google Maps SDK.
 */
export function isValidMapCoord(
  latitude: number | null | undefined,
  longitude: number | null | undefined
): boolean {
  if (typeof latitude !== "number" || typeof longitude !== "number") return false;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return false;
  if (latitude < -90 || latitude > 90) return false;
  if (longitude < -180 || longitude > 180) return false;
  return true;
}
