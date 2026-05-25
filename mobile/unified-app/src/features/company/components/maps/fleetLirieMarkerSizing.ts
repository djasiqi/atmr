/**
 * Largeur cible du pin sur la carte.
 * Les PNG sont générés à cette taille en pixels : Google Maps utilise souvent
 * la taille intrinsèque du bitmap quand width/height sur `icon` sont ignorés.
 */
export const LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX = 48;

/** = DISPLAY — PNG 1:1 ; regénérer via `npm run sync:fleet-markers`. */
export const LIRIE_DRIVER_MARKER_PNG_WIDTH_PX = LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX;

/** Pastille de regroupement (chauffeurs dans la même zone). */
export const LIRIE_CLUSTER_MARKER_DISPLAY_WIDTH_PX = LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX;

/** Hauteur de la pastille compteur sur l’icône PNG (lisible sur carte). */
export const LIRIE_CLUSTER_COUNT_BADGE_HEIGHT_PX = 24;

/** Alias partagé (évite d’importer fleetNativeMarkerImage depuis la légende / UI). */
export const FLEET_NATIVE_DRIVER_MARKER_SIZE_PX = LIRIE_DRIVER_MARKER_DISPLAY_WIDTH_PX;
