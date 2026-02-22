/**
 * Service de validation et parsing des deep links ATMR.
 *
 * Valide les deep links pour éviter les injections et open redirects.
 * Format attendu: atmr://{type}/{id}
 * Types supportés: booking, chat, dispatch
 */

import { getLogger } from "@/utils/logger";

const log = getLogger("DeepLinks");
export interface DeepLinkValidationResult {
  valid: boolean;
  type?: string;
  /** Sous-type pour formats complexes (ex: "message" | "thread" pour chat) */
  subType?: string;
  id?: number;
  error?: string;
}

/**
 * Valide et parse un deep link ATMR.
 *
 * @param url - URL du deep link (ex: "atmr://booking/123")
 * @returns Résultat de validation avec type et ID si valide
 *
 * @example
 * ```typescript
 * const result = validateDeepLink("atmr://booking/123");
 * if (result.valid) {
 *   console.log(`Type: ${result.type}, ID: ${result.id}`);
 * }
 * ```
 */
export function validateDeepLink(url: string): DeepLinkValidationResult {
  // Vérifier que l'URL commence par le schéma attendu
  if (!url || typeof url !== "string") {
    return {
      valid: false,
      error: "URL invalide ou vide",
    };
  }

  if (!url.startsWith("atmr://")) {
    return {
      valid: false,
      error: "Le deep link doit commencer par 'atmr://'",
    };
  }

  // Extraire le chemin après "atmr://"
  const path = url.replace("atmr://", "").trim();
  if (!path) {
    return {
      valid: false,
      error: "Chemin manquant après 'atmr://'",
    };
  }

  // Parser le chemin avec regex stricte
  // Formats supportés:
  // - {type}/{id} (ex: booking/123)
  // - {type}/{subtype}/{id} (ex: chat/message/456, dispatch/run/789)
  // - {type} (ex: bookings - sans ID)
  const simpleMatch = path.match(/^([a-z]+)\/(\d+)$/i);
  const complexMatch = path.match(/^([a-z]+)\/([a-z]+)\/(\d+)$/i);
  const listMatch = path.match(/^([a-z]+)$/i);

  let type: string | undefined;
  let subType: string | undefined;
  let idStr: string | undefined;

  if (simpleMatch) {
    // Format: booking/123
    [, type, idStr] = simpleMatch;
  } else if (complexMatch) {
    // Format: chat/message/456, chat/thread/789, dispatch/run/123
    const [, mainType, sub, id] = complexMatch;
    type = mainType;
    subType = sub?.toLowerCase();
    idStr = id;
  } else if (listMatch) {
    // Format: bookings (sans ID)
    [, type] = listMatch;
    // Pas d'ID pour les listes
    idStr = undefined;
  } else if (/^chat(\/(thread|message)(\/.+)?)?$/i.test(path)) {
    // Garde robustesse: chat/thread/team, chat/thread/abc, chat/thread, chat/message malformé
    // → fallback atmr://chat (évite routes cassées sur payload ancien/malformé)
    type = "chat";
    subType = undefined;
    idStr = undefined;
  } else {
    return {
      valid: false,
      error: "Format invalide. Attendu: atmr://{type}/{id} ou atmr://{type}/{subtype}/{id}",
    };
  }

  // Valider le type (whitelist)
  const validTypes = ["booking", "bookings", "chat", "dispatch"];
  if (!type || !validTypes.includes(type.toLowerCase())) {
    return {
      valid: false,
      error: `Type '${type}' non supporté. Types valides: ${validTypes.join(", ")}`,
    };
  }

  // Si pas d'ID (cas "bookings"), retourner valide sans ID
  if (!idStr) {
    return {
      valid: true,
      type: type.toLowerCase(),
    };
  }

  // Valider et parser l'ID
  const id = parseInt(idStr, 10);
  if (isNaN(id) || id <= 0 || !Number.isInteger(id)) {
    // Garde robustesse chat: tid "team" ou non numérique → fallback atmr://chat
    if (type === "chat") {
      return { valid: true, type: "chat" };
    }
    return {
      valid: false,
      error: `ID invalide: '${idStr}'. Doit être un entier positif`,
    };
  }

  // Vérifier que l'ID ne dépasse pas une limite raisonnable (évite les attaques)
  const MAX_ID = 2147483647; // Max int32
  if (id > MAX_ID) {
    return {
      valid: false,
      error: `ID trop grand: ${id}. Maximum: ${MAX_ID}`,
    };
  }

  return {
    valid: true,
    type: type.toLowerCase(),
    ...(subType && { subType }),
    id,
  };
}

/**
 * Valide un deep link et retourne une URL sécurisée pour navigation.
 *
 * @param url - URL du deep link à valider
 * @returns URL validée ou null si invalide
 */
export function getValidatedDeepLink(url: string): string | null {
  const result = validateDeepLink(url);
  if (!result.valid || !result.type || !result.id) {
    return null;
  }
  return `atmr://${result.type}/${result.id}`;
}

