import type { BootLottieAsset } from "./bootLottieAssets";
import { bootLottieAssets } from "./bootLottieAssets";

/** Dimensions de composition Lottie (px) — alignées sur les exports Jitter. */
const COMPOSITION_SIZE: Partial<Record<BootLottieAsset, { w: number; h: number }>> = {
  [bootLottieAssets.androidMedium]: { w: 700, h: 840 },
  [bootLottieAssets.androidCompact]: { w: 412, h: 917 },
  [bootLottieAssets.iphone1314]: { w: 700, h: 840 },
  [bootLottieAssets.iphone1617ProMax]: { w: 700, h: 840 },
  [bootLottieAssets.iphone16Plus]: { w: 700, h: 840 },
  [bootLottieAssets.iphone17]: { w: 700, h: 840 },
  [bootLottieAssets.iphoneAir]: { w: 700, h: 840 },
  [bootLottieAssets.iphone13Mini]: { w: 700, h: 840 },
  [bootLottieAssets.iphoneSE]: { w: 700, h: 840 },
  [bootLottieAssets.p1080]: { w: 700, h: 840 },
  [bootLottieAssets.p720]: { w: 700, h: 840 },
};

const DEFAULT_COMPOSITION = { w: 700, h: 840 };

export function compositionSizeForSource(source: BootLottieAsset): { w: number; h: number } {
  return COMPOSITION_SIZE[source] ?? DEFAULT_COMPOSITION;
}

/**
 * Taille d'affichage **plein écran** — indépendante du téléphone.
 *
 * Avant : on contraignait à 92 % × 52 % pour éviter le carré blanc visible,
 * mais cela générait un encart blanc au centre sur certains écrans (iPhone 12, etc.).
 * Désormais le Lottie occupe l'intégralité de la fenêtre ; le composant natif
 * utilise `resizeMode="cover"` pour remplir parfaitement quel que soit le ratio.
 *
 * Le paramètre `source` est conservé pour compatibilité (logs / tests / futurs ajustements).
 */
export function computeBootLottieDisplaySize(
  screenWidth: number,
  screenHeight: number,
  _source: BootLottieAsset
): { width: number; height: number } {
  return {
    width: Math.max(1, Math.round(screenWidth)),
    height: Math.max(1, Math.round(screenHeight)),
  };
}
