/**
 * Service de validation et parsing des deep links ATMR.
 *
 * Valide les deep links pour éviter les injections et open redirects.
 * Format attendu: atmr://{type}/{id}
 * Types supportés: booking, chat, dispatch
 */

export interface DeepLinkValidationResult {
  valid: boolean;
  type?: string;
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
  let idStr: string | undefined;

  if (simpleMatch) {
    // Format: booking/123
    [, type, idStr] = simpleMatch;
  } else if (complexMatch) {
    // Format: chat/message/456 ou dispatch/run/789
    const [, mainType, subType, id] = complexMatch;
    // Pour les formats complexes, on retourne le type principal et l'ID
    type = mainType;
    idStr = id;
  } else if (listMatch) {
    // Format: bookings (sans ID)
    [, type] = listMatch;
    // Pas d'ID pour les listes
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

