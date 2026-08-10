/**
 * Comportement de l’intro Lottie au démarrage.
 *
 * Exigence produit : intro complète au premier lancement après installation ;
 * les cold starts suivants sautent le Lottie (~3 s artificiels) pour un accès
 * immédiat à l’outil opérationnel.
 */
export const BOOT_LOTTIE_EVERY_COLD_START = false;

/** Si true, l’intro est jouée une seule fois par installation. */
export const BOOT_LOTTIE_FIRST_LAUNCH_ONLY = !BOOT_LOTTIE_EVERY_COLD_START;
