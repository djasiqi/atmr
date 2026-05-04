import type { BootLottieAsset } from "./bootLottieAssets";
import { bootLottieAssets } from "./bootLottieAssets";

/**
 * Cartographie des codes machine iOS (sysctl hw.machine / expo-device modelId)
 * vers les JSON du dossier assets/lottie/boot.
 *
 * Source des identifiants : liste communautaire (ex. adamawolf) + fiches AppleDB / Apple.
 * À étendre quand de nouveaux modèles sortent.
 */
const IPHONE_MODEL_ID: Record<string, BootLottieAsset> = {
  // iPhone SE
  "iphone8,4": bootLottieAssets.iphoneSE,
  "iphone12,8": bootLottieAssets.iphoneSE,
  "iphone14,6": bootLottieAssets.iphoneSE,

  // mini (12 mini → fichiers « iPhone-13-mini » comme gabarit compact)
  "iphone13,1": bootLottieAssets.iphone13Mini,
  "iphone14,4": bootLottieAssets.iphone13Mini,

  // iPhone Air (2025+)
  "iphone18,4": bootLottieAssets.iphoneAir,

  // Pro Max — fichier commun 16/17 Pro Max
  "iphone11,6": bootLottieAssets.iphone1617ProMax,
  "iphone12,5": bootLottieAssets.iphone1617ProMax,
  "iphone13,4": bootLottieAssets.iphone1617ProMax,
  "iphone14,3": bootLottieAssets.iphone1617ProMax,
  "iphone15,3": bootLottieAssets.iphone1617ProMax,
  "iphone16,2": bootLottieAssets.iphone1617ProMax,
  "iphone17,2": bootLottieAssets.iphone1617ProMax,
  "iphone18,2": bootLottieAssets.iphone1617ProMax,

  // Plus (pas Pro Max)
  "iphone14,8": bootLottieAssets.iphone16Plus,
  "iphone15,5": bootLottieAssets.iphone16Plus,
  "iphone17,4": bootLottieAssets.iphone16Plus,

  // iPhone 17 « standard »
  "iphone18,3": bootLottieAssets.iphone17,

  // iPhone 16 / 15 / 14 « standard » — même fichier générique 13–14
  "iphone14,7": bootLottieAssets.iphone1314,
  "iphone15,4": bootLottieAssets.iphone1314,
  "iphone17,3": bootLottieAssets.iphone1314,

  // Pro (taille 6,1" typique) — pas de JSON dédié : générique
  "iphone13,3": bootLottieAssets.iphone1314,
  "iphone14,2": bootLottieAssets.iphone1314,
  "iphone15,2": bootLottieAssets.iphone1314,
  "iphone16,1": bootLottieAssets.iphone1314,
  "iphone17,1": bootLottieAssets.iphone1314,
  "iphone18,1": bootLottieAssets.iphone1314,

  // iPhone 16e — gabarit proche des autres 6,1"
  "iphone17,5": bootLottieAssets.iphone1314,

  // iPhone 12 / 13 base
  "iphone13,2": bootLottieAssets.iphone1314,
  "iphone13,5": bootLottieAssets.iphone1314,
  "iphone14,5": bootLottieAssets.iphone1314,
};

/**
 * Résolution prioritaire par modelId quand disponible (simulateur / device réel).
 */
export function resolveBootLottieByIosModelId(modelId: string | null | undefined): BootLottieAsset | null {
  if (!modelId || typeof modelId !== "string") {
    return null;
  }
  const key = modelId.trim().toLowerCase();
  return IPHONE_MODEL_ID[key] ?? null;
}
