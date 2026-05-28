import { Asset } from "expo-asset";
import * as Sentry from "@sentry/react-native";

export type MetroAssetSource = {
  uri: string;
  width?: number;
  height?: number;
  scale?: number;
};

const ASSET_RESOLVE_CACHE = new Map<number, MetroAssetSource>();
const ASSET_DOWNLOAD_TIMEOUT_MS = 500;

function pickAssetUri(asset: Asset): string | null {
  const local = asset.localUri?.trim();
  if (local) return local;
  const remote = asset.uri?.trim();
  if (remote) return remote;
  return null;
}

function metroSourceFromAsset(asset: Asset): MetroAssetSource | null {
  const uri = pickAssetUri(asset);
  if (!uri) return null;
  const width = asset.width ?? undefined;
  const height = asset.height ?? undefined;
  return {
    uri,
    ...(typeof width === "number" && width > 0 ? { width } : {}),
    ...(typeof height === "number" && height > 0 ? { height } : {}),
  };
}

function reportAssetResolveFailed(moduleId: number, reason: string): void {
  if (__DEV__) {
    console.warn("[fleet-map] asset resolve failed", { moduleId, reason });
  }
  Sentry.addBreadcrumb({
    category: "fleet_map",
    message: "fleet_map.asset_resolve_failed",
    level: "warning",
    data: { moduleId, reason },
  });
}

/** Résolution sûre des `require()` PNG via expo-asset — ne lance jamais. */
export function resolveMetroAssetSource(moduleId: number): MetroAssetSource | null {
  const cached = ASSET_RESOLVE_CACHE.get(moduleId);
  if (cached) return cached;

  try {
    const asset = Asset.fromModule(moduleId);
    const resolved = metroSourceFromAsset(asset);
    if (resolved) {
      ASSET_RESOLVE_CACHE.set(moduleId, resolved);
      return resolved;
    }
    reportAssetResolveFailed(moduleId, "empty_uri");
    return null;
  } catch (error) {
    const reason = error instanceof Error ? error.message : "resolve_error";
    reportAssetResolveFailed(moduleId, reason);
    return null;
  }
}

async function downloadAssetWithTimeout(asset: Asset): Promise<void> {
  await Promise.race([
    asset.downloadAsync(),
    new Promise<void>((_, reject) => {
      setTimeout(() => reject(new Error("asset_download_timeout")), ASSET_DOWNLOAD_TIMEOUT_MS);
    }),
  ]);
}

/** Résolution async (OTA) avec téléchargement local si uri absente — ne lance jamais. */
export async function resolveMetroAssetSourceAsync(
  moduleId: number
): Promise<MetroAssetSource | null> {
  const sync = resolveMetroAssetSource(moduleId);
  if (sync) return sync;

  try {
    const asset = Asset.fromModule(moduleId);
    if (!pickAssetUri(asset)) {
      await downloadAssetWithTimeout(asset);
    }
    const resolved = metroSourceFromAsset(asset);
    if (resolved) {
      ASSET_RESOLVE_CACHE.set(moduleId, resolved);
      return resolved;
    }
    reportAssetResolveFailed(moduleId, "empty_uri_after_download");
    return null;
  } catch (error) {
    const reason = error instanceof Error ? error.message : "async_resolve_error";
    reportAssetResolveFailed(moduleId, reason);
    return null;
  }
}

export function clearMetroAssetResolveCacheForTests(): void {
  ASSET_RESOLVE_CACHE.clear();
}
