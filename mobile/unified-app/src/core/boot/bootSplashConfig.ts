/**
 * Comportement de l’intro Lottie au démarrage.
 *
 * Exigence produit actuelle : rejouer l’intro à chaque cold start
 * (après fermeture complète / app kill).
 */
export const BOOT_LOTTIE_EVERY_COLD_START = true;

/** Si true, l’intro est jouée une seule fois par installation. */
export const BOOT_LOTTIE_FIRST_LAUNCH_ONLY = !BOOT_LOTTIE_EVERY_COLD_START;
