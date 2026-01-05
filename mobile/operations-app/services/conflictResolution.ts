// services/conflictResolution.ts
/**
 * Service de détection et résolution de conflits lors du resync.
 * Gère les cas où les données locales ont été modifiées pendant que l'app était hors ligne.
 */

import type { Booking, Message } from "./api";

export type ConflictResolutionStrategy =
  | "server-wins"
  | "client-wins"
  | "merge-intelligent"
  | "timestamp-based";

export interface Conflict {
  id: number | string;
  type: "booking" | "message";
  localData: any;
  serverData: any;
  conflictingFields: string[];
  resolution: "server-wins" | "client-wins" | "merged" | "timestamp-based";
  resolvedData: any;
}

export interface ConflictResolutionResult<T> {
  resolved: T[];
  conflicts: Conflict[];
  hasConflicts: boolean;
}

// Champs critiques pour les bookings (ne peuvent pas être modifiés localement sans sync)
const CRITICAL_BOOKING_FIELDS = [
  "status",
  "driver_id",
  "scheduled_time",
  "pickup_location",
  "dropoff_location",
] as const;

// Champs qui peuvent être modifiés localement (notes, etc.)
const LOCAL_EDITABLE_FIELDS = ["notes", "notes_medical"] as const;

/**
 * Détecte les conflits entre données locales et serveur.
 */
export function detectConflict<T extends { id: number | string }>(
  local: T | undefined,
  server: T,
  type: "booking" | "message"
): Conflict | null {
  if (!local) {
    // Pas de données locales, pas de conflit
    return null;
  }

  if (local.id !== server.id) {
    // IDs différents, pas le même élément
    return null;
  }

  const conflictingFields: string[] = [];
  const localKeys = Object.keys(local);
  const serverKeys = Object.keys(server);

  // Comparer tous les champs
  const allKeys = new Set([...localKeys, ...serverKeys]);

  for (const key of allKeys) {
    // Ignorer les champs internes/metadata
    if (key.startsWith("_") || key === "id") {
      continue;
    }

    const localValue = (local as any)[key];
    const serverValue = (server as any)[key];

    // Comparaison profonde pour les objets/tableaux
    if (JSON.stringify(localValue) !== JSON.stringify(serverValue)) {
      conflictingFields.push(key);
    }
  }

  if (conflictingFields.length === 0) {
    // Pas de conflit
    return null;
  }

  return {
    id: local.id,
    type,
    localData: local,
    serverData: server,
    conflictingFields,
    resolution: "server-wins", // Sera déterminé par resolveConflict
    resolvedData: server, // Temporaire, sera remplacé par resolveConflict
  };
}

/**
 * Résout un conflit selon la stratégie spécifiée.
 */
export function resolveConflict<T extends { id: number | string }>(
  conflict: Conflict,
  strategy: ConflictResolutionStrategy = "server-wins"
): T {
  switch (strategy) {
    case "server-wins":
      // Les données serveur remplacent toujours les données locales
      return conflict.serverData as T;

    case "client-wins":
      // Les données locales sont conservées
      // Note: Dans un système réel, il faudrait sync vers le serveur
      return conflict.localData as T;

    case "merge-intelligent":
      return mergeIntelligent(conflict) as T;

    case "timestamp-based":
      // Utiliser les timestamps si disponibles, sinon fallback sur server-wins
      const localTimestamp = (conflict.localData as any)?.updated_at || (conflict.localData as any)?.modified_at;
      const serverTimestamp = (conflict.serverData as any)?.updated_at || (conflict.serverData as any)?.modified_at;
      
      if (localTimestamp && serverTimestamp) {
        const localTime = new Date(localTimestamp).getTime();
        const serverTime = new Date(serverTimestamp).getTime();
        return (localTime > serverTime ? conflict.localData : conflict.serverData) as T;
      }
      
      // Fallback sur server-wins si pas de timestamps
      return conflict.serverData as T;

    default:
      // Par défaut, server-wins
      return conflict.serverData as T;
  }
}

/**
 * Fusion intelligente des données locales et serveur.
 * Préserve les modifications locales si non conflictuelles, utilise les données serveur pour les champs critiques.
 */
function mergeIntelligent(conflict: Conflict): any {
  const { localData, serverData, conflictingFields, type } = conflict;

  if (type === "message") {
    // Pour les messages, toujours server-wins (append-only)
    return serverData;
  }

  // Pour les bookings, merge intelligent
  const merged = { ...serverData };

  for (const field of conflictingFields) {
    const isCritical = CRITICAL_BOOKING_FIELDS.includes(field as any);
    const isLocalEditable = LOCAL_EDITABLE_FIELDS.includes(field as any);

    if (isCritical) {
      // Champs critiques : toujours utiliser les données serveur
      merged[field] = (serverData as any)[field];
    } else if (isLocalEditable) {
      // Champs éditables localement : préserver les modifications locales si présentes
      const localValue = (localData as any)[field];
      if (localValue !== undefined && localValue !== null && localValue !== "") {
        merged[field] = localValue;
      } else {
        merged[field] = (serverData as any)[field];
      }
    } else {
      // Autres champs : utiliser les données serveur (source de vérité)
      merged[field] = (serverData as any)[field];
    }
  }

  return merged;
}

/**
 * Résout les conflits pour une liste de données.
 */
export function resolveConflicts<T extends { id: number | string }>(
  localData: T[],
  serverData: T[],
  type: "booking" | "message",
  strategy: ConflictResolutionStrategy = "server-wins"
): ConflictResolutionResult<T> {
  const conflicts: Conflict[] = [];
  const resolved: T[] = [];
  const localMap = new Map<number | string, T>();
  
  // Créer un map des données locales par ID
  for (const item of localData) {
    localMap.set(item.id, item);
  }

  // Traiter chaque élément serveur
  for (const serverItem of serverData) {
    const localItem = localMap.get(serverItem.id);
    
    if (localItem) {
      // Élément existe localement, détecter les conflits
      const conflict = detectConflict(localItem, serverItem, type);
      
      if (conflict) {
        // Conflit détecté, résoudre
        const resolvedItem = resolveConflict<T>(conflict, strategy);
        conflict.resolvedData = resolvedItem;
        // Déterminer la résolution finale (timestamp-based devient server-wins ou client-wins selon le résultat)
        if (strategy === "merge-intelligent") {
          conflict.resolution = "merged";
        } else if (strategy === "timestamp-based") {
          // Si timestamp-based, déterminer si c'est server-wins ou client-wins selon le résultat
          const localTimestamp = (conflict.localData as any)?.updated_at || (conflict.localData as any)?.modified_at;
          const serverTimestamp = (conflict.serverData as any)?.updated_at || (conflict.serverData as any)?.modified_at;
          if (localTimestamp && serverTimestamp) {
            const localTime = new Date(localTimestamp).getTime();
            const serverTime = new Date(serverTimestamp).getTime();
            conflict.resolution = localTime > serverTime ? "client-wins" : "server-wins";
          } else {
            conflict.resolution = "server-wins";
          }
        } else {
          conflict.resolution = strategy as "server-wins" | "client-wins";
        }
        conflicts.push(conflict);
        resolved.push(resolvedItem);
      } else {
        // Pas de conflit, utiliser les données serveur
        resolved.push(serverItem);
      }
    } else {
      // Nouvel élément du serveur, pas de conflit
      resolved.push(serverItem);
    }
  }

  // Ajouter les éléments locaux qui n'existent pas sur le serveur (supprimés côté serveur)
  // Pour les bookings, on les ignore (le serveur est la source de vérité)
  // Pour les messages, on les ignore aussi (append-only)

  return {
    resolved,
    conflicts,
    hasConflicts: conflicts.length > 0,
  };
}

/**
 * Résout les conflits pour les bookings avec stratégie merge-intelligent par défaut.
 */
export function resolveBookingConflicts(
  localBookings: Booking[],
  serverBookings: Booking[]
): ConflictResolutionResult<Booking> {
  return resolveConflicts(
    localBookings,
    serverBookings,
    "booking",
    "merge-intelligent"
  );
}

/**
 * Résout les conflits pour les messages avec stratégie server-wins par défaut.
 */
export function resolveMessageConflicts(
  localMessages: Message[],
  serverMessages: Message[]
): ConflictResolutionResult<Message> {
  return resolveConflicts(
    localMessages,
    serverMessages,
    "message",
    "server-wins"
  );
}

