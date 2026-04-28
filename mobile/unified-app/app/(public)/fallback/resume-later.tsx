import { Pressable, Text, View } from "react-native";
import { useRouter } from "expo-router";

export default function ResumeLaterScreen() {
  const router = useRouter();
  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 24, gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "800", color: "#0f172a" }}>
        Reprendre plus tard
      </Text>
      <Text style={{ color: "#475569" }}>
        Le contexte de votre session n&apos;est pas pret. Vous pourrez reprendre dans quelques instants.
      </Text>
      <Pressable onPress={() => router.replace("/(public)" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Retour accueil</Text>
      </Pressable>
      <Pressable onPress={() => router.replace("/(public)/login" as any)}>
        <Text style={{ color: "#0a7ea4", fontWeight: "700" }}>Me reconnecter</Text>
      </Pressable>
    </View>
  );
}
