import * as Updates from "expo-updates";

export const OTA_ASSET_LOAD_ERROR =
  "Impossible de télécharger la mise à jour. Vérifiez votre connexion (Wi‑Fi de préférence) et réessayez.";

export type OtaApplyResult = "reloaded" | "not_new" | "failed";

export function resolveOtaApplyErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.includes("Failed to load all assets")) {
    return OTA_ASSET_LOAD_ERROR;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "update_apply_failed";
}

export async function fetchAndReloadOtaUpdate(): Promise<OtaApplyResult> {
  try {
    const fetchResult = await Updates.fetchUpdateAsync();
    if (!fetchResult.isNew) {
      return "not_new";
    }
    await Updates.reloadAsync();
    return "reloaded";
  } catch {
    return "failed";
  }
}

export async function reloadPendingOtaUpdate(): Promise<OtaApplyResult> {
  try {
    await Updates.reloadAsync();
    return "reloaded";
  } catch {
    return "failed";
  }
}
