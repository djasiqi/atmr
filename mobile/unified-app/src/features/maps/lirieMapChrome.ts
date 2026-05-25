import { Platform, type ViewStyle } from "react-native";

/**
 * Hauteur réservée au logo / lien Google en bas de carte (iOS, Android, web).
 * Le conteneur parent masque cette bande via overflow + décalage carte.
 */
/** Bande rognée en bas (logo / lien Google Maps) — iOS / web. */
export const LIRIE_GOOGLE_MAP_LOGO_CLIP_PX = 64;

/** Android : logo souvent plus haut + SurfaceView au-dessus des overlays frères. */
export const LIRIE_GOOGLE_MAP_LOGO_CLIP_ANDROID_PX = 96;

/** Mention « Google » Android : coin bas-gauche, parfois hors bandeau pleine largeur. */
export const LIRIE_GOOGLE_ATTRIBUTION_ANDROID_LEFT_BLEED_PX = 32;

export const LIRIE_GOOGLE_ATTRIBUTION_ANDROID_LEFT_PATCH_WIDTH_PX = 280;

/** Repousse l’attribution Google vers la droite (mapPadding natif). */
export const LIRIE_GOOGLE_ATTRIBUTION_ANDROID_MAP_PADDING_LEFT_PX = 56;

/** Cartes mission compactes (~150px) : rognage bas sans mapPadding (évite bandes blanches). */
export const COMPACT_MISSION_MAP_GOOGLE_CLIP_ANDROID_PX = 40;
export const COMPACT_MISSION_MAP_GOOGLE_CLIP_IOS_PX = 32;

export function resolveCompactMissionMapGoogleClipPx(): number {
  return Platform.OS === "android"
    ? COMPACT_MISSION_MAP_GOOGLE_CLIP_ANDROID_PX
    : COMPACT_MISSION_MAP_GOOGLE_CLIP_IOS_PX;
}

/** MapView légèrement plus haute, rognée par le conteneur parent (`overflow: hidden`). */
export function compactMissionMapLayerStyle(
  visibleHeight: number,
  clipPx = resolveCompactMissionMapGoogleClipPx()
): ViewStyle {
  return {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    height: visibleHeight + clipPx,
  };
}

export function resolveNativeGoogleLogoClipPx(): number {
  return Platform.OS === "android"
    ? LIRIE_GOOGLE_MAP_LOGO_CLIP_ANDROID_PX
    : LIRIE_GOOGLE_MAP_LOGO_CLIP_PX;
}

/** Hauteur du bandeau opaque natif (au-dessus de la MapView). */
export function nativeGoogleAttributionMaskHeight(): number {
  const clip = resolveNativeGoogleLogoClipPx();
  return clip + (Platform.OS === "android" ? 32 : 12);
}

/** Couleur du masque bas (proche fond carte Lirie / cockpit). */
export const LIRIE_GOOGLE_MAP_ATTRIBUTION_MASK_COLOR = "#E8EDEB";

/** Conteneur visible de la carte (hauteur affichée). */
export function lirieMapClipViewportStyle(height: number): ViewStyle {
  return {
    height,
    overflow: "hidden",
    width: "100%",
  };
}

/** Carte légèrement plus haute, décalée vers le bas pour rogner le logo Google. */
export function lirieMapClipCanvasStyle(height: number): ViewStyle {
  const clip = resolveNativeGoogleLogoClipPx();
  return {
    height: height + clip,
    marginBottom: -clip,
    width: "100%",
  };
}

/** Hôte Google Maps JS (web) : étend sous le cadre pour le rognage. */
export function lirieWebMapHostStyle(): ViewStyle {
  return {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: -LIRIE_GOOGLE_MAP_LOGO_CLIP_PX,
  };
}

const LIRIE_GOOGLE_ATTRIBUTION_HIDE = `
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  max-height: 0 !important;
  overflow: hidden !important;
`;

export const LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS = `
  .liri-web-map-showcase .gmnoprint,
  .liri-web-map-showcase .gm-style-cc,
  .liri-web-map-showcase .gm-style-cc > *,
  .liri-web-map-showcase .gm-style-moc,
  .liri-web-map-showcase .liri-google-map-host .gmnoprint,
  .liri-web-map-showcase .liri-google-map-host .gm-style-cc,
  .liri-web-map-showcase .gm-style > div > div > a[href*="google"],
  .liri-web-map-showcase .gm-style > div > div > a[href*="maps"],
  .liri-web-map-showcase .gm-bundled-control,
  .liri-web-map-showcase .gm-bundled-control-on-bottom,
  .liri-web-map-showcase .gm-svpc,
  .liri-web-map-showcase button.gm-control-active,
  .liri-web-map-showcase a[href*="maps.google.com"],
  .liri-web-map-showcase a[href*="google.com/maps"],
  .liri-web-map-showcase a[title*="Google"],
  .liri-web-map-showcase a[title*="Google Maps"],
  .liri-web-map-showcase a[aria-label*="Google"],
  .liri-web-map-showcase a[aria-label*="Google Maps"],
  .liri-web-map-showcase a[aria-label*="Ouvrir cette zone dans Google Maps"],
  .liri-web-map-showcase img[alt="Google"],
  .liri-web-map-showcase img[src*="google_logo"],
  .liri-web-map-showcase img[src*="google_white"],
  .liri-web-map-showcase gmp-internal-google-attribution,
  .liri-web-map-showcase gmp-internal-google-attribution *,
  .liri-web-map-showcase [class*="google-attribution"],
  .liri-web-map-showcase [class*="GoogleAttribution"],
  .liri-web-map-showcase .watermark,
  .liri-web-map-showcase .liri-google-map-host gmp-internal-google-attribution {
    ${LIRIE_GOOGLE_ATTRIBUTION_HIDE}
  }
`;

/** @deprecated Utiliser LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS */
export const LIRIE_WEB_MAP_CHROME_CSS = LIRIE_GOOGLE_MAP_ATTRIBUTION_HIDE_CSS;

/**
 * Padding carte native minimal — repousse le logo Google hors de la zone visible.
 * Ne pas y fusionner les insets caméra (sinon bande grise en bas de la carte).
 */
export function lirieMapLogoHidePadding(): {
  top: number;
  right: number;
  bottom: number;
  left: number;
} {
  const clip = resolveNativeGoogleLogoClipPx();
  const leftPad =
    Platform.OS === "android" ? LIRIE_GOOGLE_ATTRIBUTION_ANDROID_MAP_PADDING_LEFT_PX : 0;
  return { top: 0, right: 0, left: leftPad, bottom: clip };
}

/** @deprecated Préférer `lirieMapLogoHidePadding` pour `mapPadding` ; garder les insets pour `fitToCoordinates` uniquement. */
export function lirieMapPaddingWithLogoClip(
  insets?: { top: number; right: number; bottom: number; left: number }
): { top: number; right: number; bottom: number; left: number } | undefined {
  const logoPad = lirieMapLogoHidePadding();
  if (!insets) return logoPad;
  return {
    ...insets,
    left: insets.left + logoPad.left,
    bottom: insets.bottom + logoPad.bottom,
  };
}
