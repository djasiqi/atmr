import { Asset } from "expo-asset";

export type MetroAssetSource = {
  uri: string;
  width?: number;
  height?: number;
  scale?: number;
};

/** Résolution sûre des `require()` PNG via l'API publique `expo-asset` (évite les imports internes). */
export function resolveMetroAssetSource(moduleId: number): MetroAssetSource {
  const asset = Asset.fromModule(moduleId);
  const uri = asset.uri?.trim();
  if (!uri) {
    throw new Error("[fleet-map] Impossible de resoudre l'asset PNG du marqueur");
  }

  const width = asset.width ?? undefined;
  const height = asset.height ?? undefined;

  return {
    uri,
    ...(typeof width === "number" && width > 0 ? { width } : {}),
    ...(typeof height === "number" && height > 0 ? { height } : {}),
  };
}
