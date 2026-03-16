// src/lib/push.ts
import { getLogger } from "@/utils/logger";
import Constants from "expo-constants";
import * as Notifications from "expo-notifications";
import { Platform } from "react-native";

const log = getLogger("Push");

export async function registerForPushAsync() {
  if (Constants.appOwnership === "expo") {
    log.warn("Use a Development Build (not Expo Go) to test push.");
    return null;
  }

  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', {
      name: 'default',
      importance: Notifications.AndroidImportance.MAX,
    });
  }

  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;
  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }
  if (finalStatus !== 'granted') return null;

  const token = await Notifications.getExpoPushTokenAsync();
  const preview = token?.data ? `${token.data.slice(0, 6)}...${token.data.slice(-4)}` : "n/a";
  log.info("Expo push token generated", { token_preview: preview });
  return token.data;
}
