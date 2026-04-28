import React, { type ReactNode } from "react";
import { Platform } from "react-native";
import { useFonts, type FontSource } from "expo-font";
import Ionicons from "@expo/vector-icons/Ionicons";

/**
 * Sur le web, @expo/vector-icons charge Ionicons en async (fontfaceobserver, 6s max).
 * Sans préchargement, la première page peut rejeter la promesse et polluer la console.
 * Sur iOS/Android, pas de hook useFonts ici (évite une map vide).
 */
function WebIoniconsFontGateInner({ children }: { children: ReactNode }) {
  const [loaded] = useFonts(Ionicons.font as Record<string, FontSource>);
  if (!loaded) return null;
  return <>{children}</>;
}

export function WebIoniconsFontGate({ children }: { children: ReactNode }) {
  if (Platform.OS !== "web") {
    return <>{children}</>;
  }
  return <WebIoniconsFontGateInner>{children}</WebIoniconsFontGateInner>;
}
