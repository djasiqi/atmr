import { Pressable, Text, View } from "react-native";
import * as Linking from "expo-linking";

const IOS_STORE_URL = "https://apps.apple.com";
const ANDROID_STORE_URL = "https://play.google.com/store";

export default function InstallAppScreen() {
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "800", color: "#0f172a" }}>
        Installer l&apos;application
      </Text>
      <Text style={{ color: "#475569" }}>
        Cette action requiert l&apos;application mobile Lirie.
      </Text>
      <Pressable onPress={() => void Linking.openURL(IOS_STORE_URL)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>App Store</Text>
      </Pressable>
      <Pressable onPress={() => void Linking.openURL(ANDROID_STORE_URL)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Google Play</Text>
      </Pressable>
    </View>
  );
}
