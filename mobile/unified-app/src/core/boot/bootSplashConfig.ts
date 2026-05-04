/**
 * Comportement de l’intro Lottie au démarrage.
 *
 * Par défaut : une seule fois par installation (`AsyncStorage`).
 * Pour rejouer l’animation à **chaque** cold start (ancien comportement) :
 * `EXPO_PUBLIC_BOOT_LOTTIE_EVERY_COLD_START=true`
 */
export const BOOT_LOTTIE_EVERY_COLD_START =
  process.env.EXPO_PUBLIC_BOOT_LOTTIE_EVERY_COLD_START === "true" ||
  process.env.EXPO_PUBLIC_BOOT_LOTTIE_EVERY_COLD_START === "1";

/** Intro jouée une seule fois ; si false, pas de persistance (rejoue à chaque lancement). */
export const BOOT_LOTTIE_FIRST_LAUNCH_ONLY = !BOOT_LOTTIE_EVERY_COLD_START;
