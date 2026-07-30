/** Helpers purs pour URLs audio chat (sans dépendance native FileSystem). */

export function isPrivateChatUploadUrl(uri: string): boolean {
  const value = uri.trim();
  if (!value) return false;
  if (/\/uploads\/chat\//i.test(value)) return true;
  return value.startsWith("/uploads/chat/") || value.startsWith("uploads/chat/");
}

export function isNumericMessageId(messageId: string | number | null | undefined): boolean {
  if (messageId == null) return false;
  const s = String(messageId).trim();
  return /^\d+$/.test(s);
}

export function inferChatAudioCacheExtension(uri: string): string {
  const clean = uri.split("?")[0] ?? uri;
  const ext = clean.split(".").pop()?.toLowerCase();
  if (ext && ["m4a", "mp3", "wav", "aac", "caf", "3gp", "webm", "ogg"].includes(ext)) {
    return ext;
  }
  return "m4a";
}
