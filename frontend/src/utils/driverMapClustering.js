/**
 * Tri-state clustering carte chauffeurs (S1.5).
 *
 * REACT_APP_ENABLE_DRIVER_CLUSTERING :
 * - 'true'  → toujours actif
 * - 'false' → toujours désactivé (override absolu)
 * - absent / autre → auto si driverCount > seuil (défaut 50)
 */

const DEFAULT_THRESHOLD = 50;

export function resolveDriverClusteringThreshold(env = process.env) {
  const raw = env.REACT_APP_DRIVER_CLUSTERING_THRESHOLD;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_THRESHOLD;
}

export function resolveDriverClusteringEnabled(driverCount, env = process.env) {
  const mode = String(env.REACT_APP_ENABLE_DRIVER_CLUSTERING ?? '').trim().toLowerCase();
  if (mode === 'true') return true;
  if (mode === 'false') return false;
  const threshold = resolveDriverClusteringThreshold(env);
  return Number(driverCount) > threshold;
}
