/**
 * Animations Lottie du démarrage à froid (cold start).
 * Fichiers dans assets/lottie/boot/ — voir resolveBootLottieSource pour le mapping appareil.
 */

export const bootLottieAssets = {
  iphoneAir: require("../../../assets/lottie/boot/iPhone-Air-Logo-Move.json"),
  iphone17: require("../../../assets/lottie/boot/iPhone-17-Logo-Move.json"),
  iphone1617ProMax: require("../../../assets/lottie/boot/iPhone-16-17-Pro-Max-Logo-Move.json"),
  iphone16Plus: require("../../../assets/lottie/boot/iPhone-16-Plus-Logo-Move.json"),
  iphone1314: require("../../../assets/lottie/boot/iPhone-13-14-Logo-Move.json"),
  iphone13Mini: require("../../../assets/lottie/boot/iPhone-13-mini.json"),
  iphoneSE: require("../../../assets/lottie/boot/iPhone-SE-Logo-Move.json"),
  androidMedium: require("../../../assets/lottie/boot/Android-Medium-Logo-Move.json"),
  androidCompact: require("../../../assets/lottie/boot/Android-Compact-Logo-Move.json"),
  p1080: require("../../../assets/lottie/boot/1080p-Logo-Move.json"),
  p720: require("../../../assets/lottie/boot/720-Logo-Move.json"),
} as const;

export type BootLottieAsset = (typeof bootLottieAssets)[keyof typeof bootLottieAssets];
