import * as Device from "expo-device";
import { Platform } from "react-native";
import { bootLottieAssets, type BootLottieAsset } from "./bootLottieAssets";
import { resolveBootLottieByIosModelId } from "./iosBootLottieMapping";

const ANDROID_MEDIUM_MIN_WIDTH_DP = 420;

function resolveIosBootLottieFromMarketingName(
  rawName: string,
  screenWidth: number,
  screenHeight: number
): BootLottieAsset {
  const w = Math.min(screenWidth, screenHeight);
  const h = Math.max(screenWidth, screenHeight);
  const name = rawName.toLowerCase().trim();

  // SE — toutes générations (« iPhone SE », « iPhone SE (2nd generation) », …)
  if (name.includes("iphone se") || name === "iphonese") {
    return bootLottieAssets.iphoneSE;
  }

  // mini — « iPhone 12 mini », « iPhone 13 mini », …
  if (/\bmini\b/.test(name)) {
    return bootLottieAssets.iphone13Mini;
  }

  // Air — nom commercial exact attendu
  if (name.includes("iphone air")) {
    return bootLottieAssets.iphoneAir;
  }

  // Pro Max avant « Plus » pour éviter toute ambiguïté
  if (name.includes("pro max")) {
    return bootLottieAssets.iphone1617ProMax;
  }

  // Plus — « iPhone 14 Plus », « iPhone 16 Plus », pas les « Pro »
  if (/\bplus\b/.test(name) && !name.includes("pro")) {
    return bootLottieAssets.iphone16Plus;
  }

  // iPhone 17 sans qualificatif Pro / Plus / mini / Air
  const isIphone17Base =
    /^iphone\s*17$/i.test(rawName.trim()) ||
    (/iphone\s*17/i.test(rawName) && !/pro|plus|mini|air/i.test(rawName));
  if (isIphone17Base) {
    return bootLottieAssets.iphone17;
  }

  if (h <= 667 && w <= 380) {
    return bootLottieAssets.iphoneSE;
  }
  if (h <= 812 && w <= 380) {
    return bootLottieAssets.iphone13Mini;
  }

  return bootLottieAssets.iphone1314;
}

/**
 * Choisit le JSON Lottie adapté à la plateforme / modèle / taille d’écran.
 * iOS : `modelId` (machine) en priorité, puis nom commercial, puis heuristiques de taille.
 */
export function resolveBootLottieSource(screenWidth: number, screenHeight: number): BootLottieAsset {
  const w = Math.min(screenWidth, screenHeight);
  const h = Math.max(screenWidth, screenHeight);

  if (Platform.OS === "web") {
    return screenWidth >= 1080 ? bootLottieAssets.p1080 : bootLottieAssets.p720;
  }

  if (Platform.OS === "android") {
    return screenWidth >= ANDROID_MEDIUM_MIN_WIDTH_DP ? bootLottieAssets.androidMedium : bootLottieAssets.androidCompact;
  }

  if (Platform.OS === "ios") {
    const modelId = Device.modelId ?? null;
    const fromId = resolveBootLottieByIosModelId(modelId);
    if (fromId) {
      return fromId;
    }

    const rawName = Device.modelName ?? "";
    if (rawName.length > 0) {
      return resolveIosBootLottieFromMarketingName(rawName, screenWidth, screenHeight);
    }

    if (h <= 667 && w <= 380) {
      return bootLottieAssets.iphoneSE;
    }
    if (h <= 812 && w <= 380) {
      return bootLottieAssets.iphone13Mini;
    }
    return bootLottieAssets.iphone1314;
  }

  return bootLottieAssets.iphone1314;
}
