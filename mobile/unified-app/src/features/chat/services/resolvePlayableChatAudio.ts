/**
 * Résout une URI audio lisible localement.
 *
 * Depuis SEC-06, `/uploads/chat/*` n'est plus public (404 anonyme).
 * expo-audio ne peut donc pas streamer l'URL API directement :
 * on télécharge via `GET /messages/:id/attachment` (JWT) vers le cache.
 */
import { Platform } from "react-native";
// API legacy stable (downloadAsync / cacheDirectory) — le package racine throw en runtime.
// eslint-disable-next-line import/no-extraneous-dependencies -- fourni par le package expo
import {
  cacheDirectory,
  deleteAsync,
  downloadAsync,
  getInfoAsync,
} from "expo-file-system/legacy";

import { apiClient, getAuthAccessToken } from "../../../core/api/client";
import { resolveMediaUrl } from "../../../core/api/mediaUrl";
import {
  inferChatAudioCacheExtension,
  isNumericMessageId,
  isPrivateChatUploadUrl,
} from "./chatAudioUrl";

export {
  inferChatAudioCacheExtension,
  isNumericMessageId,
  isPrivateChatUploadUrl,
} from "./chatAudioUrl";

function getApiOrigin(): string {
  const base = String(apiClient.defaults.baseURL ?? "").trim();
  if (!base) return "";
  try {
    return new URL(base).origin;
  } catch {
    return base.replace(/\/api(?:\/v\d+)?(?:\/.*)?$/i, "").replace(/\/+$/, "");
  }
}

/**
 * Retourne une URI `file://` (ou locale) jouable par expo-audio.
 */
export async function resolvePlayableChatAudioUri(options: {
  uri: string;
  messageId?: string | number | null;
}): Promise<string> {
  const raw = String(options.uri ?? "").trim();
  if (!raw) {
    throw new Error("URI audio manquante");
  }

  if (
    raw.startsWith("file:") ||
    raw.startsWith("content:") ||
    raw.startsWith("data:")
  ) {
    return raw;
  }

  const resolved = resolveMediaUrl(raw) ?? raw;

  // Web / hors privé : laisser l'URL telle quelle.
  if (Platform.OS === "web" || !isPrivateChatUploadUrl(resolved)) {
    return resolved;
  }

  if (!isNumericMessageId(options.messageId)) {
    throw new Error("Pièce jointe audio inaccessible (message non persisté).");
  }

  const messageId = String(options.messageId).trim();
  const cacheDir = cacheDirectory;
  if (!cacheDir) {
    throw new Error("Cache fichier indisponible.");
  }

  const ext = inferChatAudioCacheExtension(resolved);
  const target = `${cacheDir}chat-audio-${messageId}.${ext}`;
  const existing = await getInfoAsync(target);
  if (existing.exists && typeof existing.size === "number" && existing.size > 0) {
    return target;
  }

  const token = getAuthAccessToken();
  if (!token) {
    throw new Error("Session expirée. Reconnectez-vous pour écouter les vocaux.");
  }

  const origin = getApiOrigin();
  const downloadUrl = origin
    ? `${origin}/api/v1/messages/${messageId}/attachment`
    : `/api/v1/messages/${messageId}/attachment`;

  const result = await downloadAsync(downloadUrl, target, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "*/*",
    },
  });

  if (result.status < 200 || result.status >= 300) {
    try {
      await deleteAsync(target, { idempotent: true });
    } catch {
      /* ignore */
    }
    throw new Error(`Téléchargement audio échoué (${result.status}).`);
  }

  return result.uri || target;
}
