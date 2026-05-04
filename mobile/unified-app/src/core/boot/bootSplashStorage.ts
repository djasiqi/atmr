import AsyncStorage from "@react-native-async-storage/async-storage";

const KEY = "@lirie/boot_lottie_intro_seen_v1";

export async function getBootLottieIntroSeen(): Promise<boolean> {
  try {
    const v = await AsyncStorage.getItem(KEY);
    return v === "1";
  } catch {
    return false;
  }
}

export async function setBootLottieIntroSeen(): Promise<void> {
  try {
    await AsyncStorage.setItem(KEY, "1");
  } catch {
    // best-effort : l’intro peut rejouer au prochain cold start
  }
}
