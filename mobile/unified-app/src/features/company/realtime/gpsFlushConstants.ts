/** Fenêtre de consolidation des updates GPS avant publication React (legacy défaut). */
export const REALTIME_FLUSH_MS = 500;
/** Âge max d'un batch en attente avant flush forcé (anti-starvation). */
export const MAX_BATCH_AGE_MS = 1_000;
/** Cible PR3 après validation terrain (override via env / bootstrap). */
export const PR3_TARGET_REALTIME_FLUSH_MS = 100;
export const PR3_TARGET_MAX_BATCH_AGE_MS = 300;
