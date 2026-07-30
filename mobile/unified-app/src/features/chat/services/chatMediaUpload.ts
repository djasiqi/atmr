import { Platform } from "react-native";
import { apiClient } from "../../../core/api/client";
import { resolveMediaUrl } from "../../../core/api/mediaUrl";

type UploadChatFileOptions = {
  uri: string;
  filename?: string;
  mimeType?: string;
};

function inferVoiceFilename(uri: string): string {
  const clean = uri.split("?")[0] ?? uri;
  const ext = clean.split(".").pop()?.toLowerCase();
  if (ext && ["m4a", "caf", "mp3", "wav", "3gp", "aac", "webm"].includes(ext)) {
    return `voice-${Date.now()}.${ext}`;
  }
  return `voice-${Date.now()}.m4a`;
}

function inferVoiceMimeType(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "mp3":
      return "audio/mpeg";
    case "wav":
      return "audio/wav";
    case "caf":
      return "audio/x-caf";
    case "3gp":
      return "audio/3gpp";
    case "aac":
      return "audio/aac";
    case "webm":
      return "audio/webm";
    case "m4a":
      return "audio/mp4";
    default:
      // Android / expo-audio HIGH_QUALITY → souvent conteneur mp4/m4a
      return "audio/mp4";
  }
}

/**
 * Upload image / PDF / audio vers `/messages/upload` et retourne l’URL publique résolue.
 */
export async function uploadChatAttachment(options: UploadChatFileOptions): Promise<string> {
  if (Platform.OS === "web") {
    throw new Error("Upload indisponible sur le web.");
  }

  const filename = options.filename?.trim() || inferVoiceFilename(options.uri);
  const mimeType = options.mimeType?.trim() || inferVoiceMimeType(filename);

  const form = new FormData();
  form.append("file", {
    uri: options.uri,
    name: filename,
    type: mimeType,
  } as unknown as Blob);

  const { data } = await apiClient.post<{ url?: string }>("/messages/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  const rawUrl = typeof data?.url === "string" ? data.url.trim() : "";
  if (!rawUrl) {
    throw new Error("Réponse upload invalide.");
  }

  return resolveMediaUrl(rawUrl) ?? rawUrl;
}
