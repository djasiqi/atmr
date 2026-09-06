/**
 * DRIVER-COLD-01 — une seule couleur et un seul logo pour toutes les
 * surfaces de démarrage (splash natif Expo, overlay JS, hold, Redirect).
 * Aucune de ces surfaces ne doit retomber sur #FFFFFF.
 *
 * `BOOT_BRAND_LOGO_WIDTH` doit rester égal à `imageWidth` du plugin
 * `expo-splash-screen` dans `app.json` (220) pour éviter un saut de taille
 * entre le splash natif et BootBrandMark. Hauteur = ratio du PNG 1078×466.
 */
export const SPLASH_BACKGROUND_COLOR = "#EAF3F1";
export const BOOT_BRAND_LOGO_WIDTH = 220;
export const BOOT_BRAND_LOGO_HEIGHT = 95;

/** Wordmark LIRIE (RGBA, fond transparent) — même asset que le splash natif. */
export const BOOT_BRAND_LOGO = require("../../../assets/images/lirie-logo-color.png");
